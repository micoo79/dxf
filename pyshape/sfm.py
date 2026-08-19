"""Inkrementális Structure-from-Motion — az "Align photos" lépés.

Menet:
  1. SIFT jellemzőpontok + párosítás (features.py)
  2. kötőpont-láncok építése
  3. induló képpár: E-mátrix (vagy sík jelenetnél homográfia) felbontása
  4. további képek: PnP RANSAC + új pontok triangulálása + időnkénti BA
  5. végső globális BA az intrinsics finomításával

Az eredmény a projektbe kerül: photo.R/C, ritka pontfelhő + megfigyelések.
A koordinátakeret ilyenkor még tetszőleges (skála nélküli) SfM-keret; a
georeferálást a georef modul végzi.
"""

import cv2
import numpy as np

from . import features as F
from .ba import BAProblem, rt_from_pose, pose_from_rt

MIN_TRI_ANGLE_DEG = 1.0
REPROJ_THRESH = 4.0


def _normalize(pts, cam):
    """Pixelekből normalizált (torzításmentes) koordináták."""
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    und = cv2.undistortPoints(p, cam.K(), cam.dist_coeffs())
    return und.reshape(-1, 2)


def _triangulate_nviews(Ps, xs):
    """DLT: Ps list of 3x4 ([R|t], normalizált koord.), xs list of (x,y)."""
    A = np.zeros((2 * len(Ps), 4))
    for i, (P, x) in enumerate(zip(Ps, xs)):
        A[2 * i] = x[0] * P[2] - P[0]
        A[2 * i + 1] = x[1] * P[2] - P[1]
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def _tri_angle_deg(X, C1, C2):
    v1 = X - C1
    v2 = X - C2
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


class _Recon:
    """Belső, épülő rekonstrukció."""

    def __init__(self, project, feats, tracks, log):
        self.prj = project
        self.feats = feats
        self.tracks = tracks
        self.log = log
        self.names = [p.name for p in project.photos]
        self.name2idx = {n: i for i, n in enumerate(self.names)}
        self.registered = {}          # photo_idx -> (R, C)
        self.track_point = {}         # track_idx -> point_idx
        self.points = []              # list of 3-vec
        self.point_track = []         # point_idx -> track_idx
        # gyors keresés: photo_name -> list[(track_idx, feat_idx)]
        self.photo_tracks = {n: [] for n in self.names}
        for ti, tr in enumerate(tracks):
            for n, fi in tr.items():
                if n in self.photo_tracks:
                    self.photo_tracks[n].append((ti, fi))

    def cam_of(self, pidx):
        return self.prj.cameras[self.prj.photos[pidx].cam_id]

    def uv_of(self, pidx, feat_idx):
        return self.feats[self.names[pidx]][0][feat_idx]

    # ------------------------------------------------------------ init pár

    def try_init_pair(self, ia, ib, matches_ab):
        camA = self.cam_of(ia); camB = self.cam_of(ib)
        pa = self.feats[self.names[ia]][0][matches_ab[:, 0]]
        pb = self.feats[self.names[ib]][0][matches_ab[:, 1]]
        na = _normalize(pa, camA)
        nb = _normalize(pb, camB)
        K = np.eye(3)

        E, maskE = cv2.findEssentialMat(na, nb, K, method=cv2.RANSAC,
                                        prob=0.999, threshold=2.0 / camA.f)
        nE = int(maskE.sum()) if maskE is not None else 0
        H, maskH = cv2.findHomography(na, nb, cv2.RANSAC, 3.0 / camA.f)
        nH = int(maskH.sum()) if maskH is not None else 0

        candidates = []
        if E is not None and nE >= 30:
            n_ok, R, t, mask = cv2.recoverPose(E, na, nb, K, mask=maskE.copy())
            if n_ok >= 20:
                candidates.append(("E", R, t.ravel(), maskE.ravel().astype(bool)))
        if H is not None and nH >= 30 and nH > 0.9 * max(nE, 1):
            try:
                _, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)
                for R, t in zip(Rs, Ts):
                    candidates.append(("H", R, t.ravel(),
                                       maskH.ravel().astype(bool)))
            except cv2.error:
                pass
        best = None
        for kind, R, t, mask in candidates:
            if np.linalg.norm(t) < 1e-9:
                continue
            t = t / np.linalg.norm(t)
            # trianguláció és minőségértékelés
            P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = np.hstack([R, t.reshape(3, 1)])
            X4 = cv2.triangulatePoints(P1, P2, na[mask].T, nb[mask].T)
            X = (X4[:3] / np.where(np.abs(X4[3]) < 1e-12, 1e-12, X4[3])).T
            z1 = X[:, 2]
            z2 = (X @ R.T + t)[:, 2]
            front = (z1 > 0) & (z2 > 0)
            if front.sum() < 20:
                continue
            C2 = -R.T @ t
            angs = np.array([_tri_angle_deg(x, np.zeros(3), C2)
                             for x in X[front][:200]])
            med_ang = np.median(angs)
            score = front.sum() * min(med_ang, 8.0)
            if best is None or score > best[0]:
                best = (score, kind, R, t, mask, med_ang, front.sum())
        if best is None:
            return None
        _, kind, R, t, mask, med_ang, nfront = best
        if med_ang < 0.5:
            return None
        self.log(f"  induló pár: {self.names[ia]} - {self.names[ib]} "
                 f"({kind}-modell, {nfront} pont, medián szög {med_ang:.1f}°)")
        return R, t, mask

    def bootstrap(self, matches):
        """Legjobb induló pár kiválasztása és inicializálás."""
        scored = sorted(matches.items(), key=lambda kv: -len(kv[1]))
        for (na_, nb_), m in scored[:12]:
            ia, ib = self.name2idx[na_], self.name2idx[nb_]
            if self.prj.photos[ia].cam_id != self.prj.photos[ib].cam_id:
                continue
            r = self.try_init_pair(ia, ib, m)
            if r is None:
                continue
            R, t, _ = r
            self.registered[ia] = (np.eye(3), np.zeros(3))
            self.registered[ib] = (R, -R.T @ t)
            self.triangulate_new()
            if len(self.points) >= 50:
                return True
            # nem sikerült elég pont — visszalépés
            self.registered.clear()
            self.track_point.clear()
            self.points = []
            self.point_track = []
        return False

    # -------------------------------------------------------- trianguláció

    def triangulate_new(self):
        """Minden még nem triangulált track, amelynek >=2 regisztrált képe van."""
        added = 0
        for ti, tr in enumerate(self.tracks):
            if ti in self.track_point:
                continue
            views = [(self.name2idx[n], fi) for n, fi in tr.items()
                     if self.name2idx[n] in self.registered]
            if len(views) < 2:
                continue
            Ps, xs, Cs = [], [], []
            for pidx, fi in views:
                R, C = self.registered[pidx]
                t = -R @ C
                Ps.append(np.hstack([R, t.reshape(3, 1)]))
                xs.append(_normalize(self.uv_of(pidx, fi),
                                     self.cam_of(pidx))[0])
                Cs.append(C)
            X = _triangulate_nviews(Ps, xs)
            if X is None:
                continue
            # ellenőrzések: mélység, szög, reprojekció
            ok = True
            for (pidx, fi), P in zip(views, Ps):
                xc = P[:, :3] @ X + P[:, 3]
                if xc[2] <= 0:
                    ok = False
                    break
                cam = self.cam_of(pidx)
                xn, yn = xc[0] / xc[2], xc[1] / xc[2]
                r2 = xn * xn + yn * yn
                d = 1 + cam.k1 * r2 + cam.k2 * r2 * r2
                u = cam.f * xn * d + cam.cx
                v = cam.f * yn * d + cam.cy
                uv = self.uv_of(pidx, fi)
                if (u - uv[0]) ** 2 + (v - uv[1]) ** 2 > REPROJ_THRESH ** 2 * 4:
                    ok = False
                    break
            if not ok:
                continue
            max_ang = 0.0
            for i in range(len(Cs)):
                for j in range(i + 1, len(Cs)):
                    max_ang = max(max_ang, _tri_angle_deg(X, Cs[i], Cs[j]))
                    if max_ang > MIN_TRI_ANGLE_DEG:
                        break
                if max_ang > MIN_TRI_ANGLE_DEG:
                    break
            if max_ang < MIN_TRI_ANGLE_DEG:
                continue
            self.track_point[ti] = len(self.points)
            self.points.append(X)
            self.point_track.append(ti)
            added += 1
        return added

    # ------------------------------------------------------------------ PnP

    def next_photo(self):
        best, best_n = None, 0
        for pidx in range(len(self.names)):
            if pidx in self.registered:
                continue
            n = sum(1 for ti, _ in self.photo_tracks[self.names[pidx]]
                    if ti in self.track_point)
            if n > best_n:
                best, best_n = pidx, n
        return best, best_n

    def register_photo(self, pidx):
        name = self.names[pidx]
        cam = self.cam_of(pidx)
        obj, img = [], []
        for ti, fi in self.photo_tracks[name]:
            if ti in self.track_point:
                obj.append(self.points[self.track_point[ti]])
                img.append(self.uv_of(pidx, fi))
        if len(obj) < 12:
            return False
        obj = np.array(obj, np.float64)
        img = np.array(img, np.float64)
        okflag, rvec, tvec, inl = cv2.solvePnPRansac(
            obj, img, cam.K(), cam.dist_coeffs(),
            reprojectionError=REPROJ_THRESH * 2, iterationsCount=300,
            flags=cv2.SOLVEPNP_SQPNP)
        if not okflag or inl is None or len(inl) < 10:
            return False
        # finomítás az inliereken
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl.ravel()], img[inl.ravel()],
                                          cam.K(), cam.dist_coeffs(),
                                          rvec, tvec)
        R, C = pose_from_rt(rvec.ravel(), tvec.ravel())
        self.registered[pidx] = (R, C)
        self.log(f"  + {name}: PnP {len(inl)}/{len(obj)} inlier")
        return True

    # ------------------------------------------------------------------- BA

    def run_ba(self, refine_intr=(), fix_first=True, max_nfev=40):
        reg = sorted(self.registered)
        if len(reg) < 2 or len(self.points) < 10:
            return
        ridx = {p: i for i, p in enumerate(reg)}
        rvecs = np.zeros((len(reg), 3)); tvecs = np.zeros((len(reg), 3))
        for p, i in ridx.items():
            R, C = self.registered[p]
            rvecs[i], tvecs[i] = rt_from_pose(R, C)
        obs_p, obs_c, obs_uv = [], [], []
        for pt_i, ti in enumerate(self.point_track):
            for n, fi in self.tracks[ti].items():
                pidx = self.name2idx[n]
                if pidx in ridx:
                    obs_p.append(pt_i)
                    obs_c.append(ridx[pidx])
                    obs_uv.append(self.uv_of(pidx, fi))
        pts = np.array(self.points)
        cam_params = [c.to_dict() for c in self.prj.cameras]
        for cp in cam_params:
            cp.pop("width"); cp.pop("height")
        prob = BAProblem(
            pts, np.array(obs_p), np.array(obs_c), np.array(obs_uv),
            rvecs, tvecs,
            [self.prj.photos[p].cam_id for p in reg],
            cam_params, None,
            refine_intrinsics=refine_intr,
            fixed_photos=(0,) if fix_first else (),
        )
        prob.solve(max_nfev=max_nfev, log=self.log)
        # visszaírás
        for p, i in ridx.items():
            R, C = pose_from_rt(prob.rvecs[i], prob.tvecs[i])
            self.registered[p] = (R, C)
        self.points = [x for x in prob.points]
        for ci, cp in enumerate(prob.cam_params):
            c = self.prj.cameras[ci]
            for k, v in cp.items():
                setattr(c, k, float(v))
        # kiugró megfigyelések/pontok szűrése
        errs = prob.reproj_errors()
        pt_err = np.zeros(len(pts)); pt_cnt = np.zeros(len(pts))
        np.add.at(pt_err, np.array(obs_p), errs)
        np.add.at(pt_cnt, np.array(obs_p), 1)
        mean_err = pt_err / np.maximum(pt_cnt, 1)
        bad = np.where(mean_err > REPROJ_THRESH * 1.5)[0]
        if len(bad):
            for b in sorted(bad, reverse=True):
                ti = self.point_track[b]
                del self.track_point[ti]
            keep = np.ones(len(self.points), bool)
            keep[bad] = False
            self.points = [x for x, k in zip(self.points, keep) if k]
            self.point_track = [t for t, k in zip(self.point_track, keep) if k]
            self.track_point = {t: i for i, t in enumerate(self.point_track)}
            self.log(f"  {len(bad)} kiugró pont törölve")


def align_photos(project, quality="high", log=print):
    """A teljes align lépés. Visszatér: statisztika dict."""
    log("Jellemzőpontok detektálása (SIFT)...")
    feats = F.detect_features(project, quality=quality, log=log)
    log("Képpárok kijelölése és illesztése...")
    pairs = F.select_pairs(project, log=log)
    matches = F.match_pairs(project, feats, pairs, log=log)
    if not matches:
        raise RuntimeError("Egyetlen képpárt sem sikerült összeilleszteni.")
    log("Kötőpont-láncok építése...")
    tracks = F.build_tracks(feats, matches, log=log)

    rec = _Recon(project, feats, tracks, log)
    log("Induló képpár keresése...")
    if not rec.bootstrap(matches):
        raise RuntimeError("Nem található megfelelő induló képpár.")
    rec.run_ba()

    since_ba = 0
    while True:
        pidx, n = rec.next_photo()
        if pidx is None or n < 12:
            break
        if not rec.register_photo(pidx):
            # sikertelen PnP — vegyük ki a jelöltek közül úgy, hogy üresre
            # állítjuk a track-listáját (többé nem jelölt)
            rec.photo_tracks[rec.names[pidx]] = []
            continue
        rec.triangulate_new()
        since_ba += 1
        if since_ba >= 3:
            rec.run_ba()
            since_ba = 0
    log("Végső kötegelt kiegyenlítés (intrinsics finomítással)...")
    rec.run_ba(refine_intr=("f", "cx", "cy", "k1", "k2"), max_nfev=80)
    rec.triangulate_new()
    rec.run_ba(refine_intr=("f", "cx", "cy", "k1", "k2"), max_nfev=40)

    # ------------------------------------------------------ projektbe írás
    for pidx, (R, C) in rec.registered.items():
        project.photos[pidx].R = R
        project.photos[pidx].C = C
    pts = np.array(rec.points) if rec.points else np.zeros((0, 3))
    obs = []
    colors = np.zeros((len(pts), 3), np.uint8)
    for pt_i, ti in enumerate(rec.point_track):
        got_color = False
        for n, fi in rec.tracks[ti].items():
            pidx = rec.name2idx[n]
            if pidx in rec.registered:
                uv = rec.uv_of(pidx, fi)
                obs.append((pt_i, pidx, uv[0], uv[1]))
                if not got_color:
                    colors[pt_i] = feats[n][2][fi]
                    got_color = True
    project.points = pts
    project.point_colors = colors
    project.observations = np.array(obs, np.float64)
    project.georeferenced = False

    n_al = len(rec.registered)
    stats = {"aligned": n_al, "total": len(project.photos),
             "points": len(pts), "observations": len(obs)}
    log(f"Align kész: {n_al}/{len(project.photos)} kép, "
        f"{len(pts)} kötőpont, {len(obs)} megfigyelés")
    for c in project.cameras:
        log(f"  kamera: f={c.f:.1f}px cx={c.cx:.1f} cy={c.cy:.1f} "
            f"k1={c.k1:.2e} k2={c.k2:.2e}")
    return stats
