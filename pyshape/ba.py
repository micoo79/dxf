"""Bundle adjustment (kötegelt kiegyenlítés) scipy ritka least_squares-szel.

Paraméterezés:
  - fényképenként: rvec (Rodrigues, 3) + tvec (3), ahol x_cam = R X + t
  - pontonként: 3 koordináta
  - kameramodellenként (opcionálisan): f, cx, cy, k1, k2 részhalmaza

Támogatott extrák:
  - rögzített fényképek / pontok (pl. GCP-k EOV-koordinátán)
  - kamerapozíció-priorok (GPS) szórással
  - megfigyelésenkénti súly (GCP-mérések felsúlyozása)
"""

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix


def rt_from_pose(R, C):
    """world->cam R és kameraközéppont C -> (rvec, tvec)."""
    rvec, _ = cv2.Rodrigues(R)
    t = -R @ C
    return rvec.ravel(), t


def pose_from_rt(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    C = -R.T @ np.asarray(tvec, float)
    return R, C


INTR_NAMES = ("f", "cx", "cy", "k1", "k2")


class BAProblem:
    def __init__(self, points, obs_pidx, obs_cidx, obs_uv,
                 photo_rvecs, photo_tvecs, photo_cam_ids, cam_params,
                 img_sizes,
                 refine_intrinsics=("f", "k1", "k2"),
                 fixed_photos=(), fixed_points=None,
                 obs_weights=None, pos_priors=None):
        """
        points: Nx3; obs_*: M hosszú tömbök; photo_*: P elemű;
        cam_params: list of dict(f,cx,cy,k1,k2); img_sizes: list of (w,h)
        fixed_points: bool tömb (N) — True = nem mozog
        pos_priors: list of (photo_idx, C_target(3), sigma) vagy None
        """
        self.points = points.astype(np.float64).copy()
        self.obs_pidx = np.asarray(obs_pidx, np.int64)
        self.obs_cidx = np.asarray(obs_cidx, np.int64)
        self.obs_uv = np.asarray(obs_uv, np.float64)
        self.rvecs = np.asarray(photo_rvecs, np.float64).copy()
        self.tvecs = np.asarray(photo_tvecs, np.float64).copy()
        self.photo_cam_ids = np.asarray(photo_cam_ids, np.int64)
        self.cam_params = [dict(c) for c in cam_params]
        self.img_sizes = img_sizes
        self.refine_intr = tuple(n for n in refine_intrinsics if n in INTR_NAMES)
        self.fixed_photos = set(int(i) for i in fixed_photos)
        n_pts = len(self.points)
        self.fixed_points = (np.zeros(n_pts, bool) if fixed_points is None
                             else np.asarray(fixed_points, bool))
        self.obs_weights = (np.ones(len(self.obs_pidx)) if obs_weights is None
                            else np.asarray(obs_weights, np.float64))
        self.pos_priors = pos_priors or []

        self.n_photos = len(self.rvecs)
        self.n_cams = len(self.cam_params)
        self.free_photos = [i for i in range(self.n_photos)
                            if i not in self.fixed_photos]
        self.free_pt_idx = np.where(~self.fixed_points)[0]
        self._pt_slot = -np.ones(n_pts, np.int64)
        self._pt_slot[self.free_pt_idx] = np.arange(len(self.free_pt_idx))
        self._ph_slot = {p: k for k, p in enumerate(self.free_photos)}

    # ---------------------------------------------------------- paraméterek

    def pack(self):
        parts = []
        for i in self.free_photos:
            parts.append(self.rvecs[i])
            parts.append(self.tvecs[i])
        parts.append(self.points[self.free_pt_idx].ravel())
        for c in self.cam_params:
            parts.append(np.array([c[n] for n in self.refine_intr]))
        return np.concatenate(parts) if parts else np.zeros(0)

    def unpack(self, x):
        o = 0
        rvecs = self.rvecs.copy(); tvecs = self.tvecs.copy()
        for i in self.free_photos:
            rvecs[i] = x[o:o + 3]; tvecs[i] = x[o + 3:o + 6]; o += 6
        pts = self.points.copy()
        nf = len(self.free_pt_idx)
        pts[self.free_pt_idx] = x[o:o + 3 * nf].reshape(-1, 3); o += 3 * nf
        cams = [dict(c) for c in self.cam_params]
        for c in cams:
            for n in self.refine_intr:
                c[n] = x[o]; o += 1
        return rvecs, tvecs, pts, cams

    # ------------------------------------------------------------ residual

    def residuals(self, x):
        rvecs, tvecs, pts, cams = self.unpack(x)
        Rs = np.zeros((self.n_photos, 3, 3))
        for i in range(self.n_photos):
            Rs[i], _ = cv2.Rodrigues(rvecs[i])
        X = pts[self.obs_pidx]
        R_o = Rs[self.obs_cidx]
        t_o = tvecs[self.obs_cidx]
        xc = np.einsum("nij,nj->ni", R_o, X) + t_o
        z = np.where(np.abs(xc[:, 2]) < 1e-9, 1e-9, xc[:, 2])
        xn = xc[:, 0] / z
        yn = xc[:, 1] / z
        r2 = xn * xn + yn * yn
        cam_arr = {n: np.array([c[n] for c in cams]) for n in INTR_NAMES}
        cid = self.photo_cam_ids[self.obs_cidx]
        f = cam_arr["f"][cid]; cx = cam_arr["cx"][cid]; cy = cam_arr["cy"][cid]
        k1 = cam_arr["k1"][cid]; k2 = cam_arr["k2"][cid]
        d = 1.0 + k1 * r2 + k2 * r2 * r2
        u = f * xn * d + cx
        v = f * yn * d + cy
        res = np.empty(2 * len(u) + 3 * len(self.pos_priors))
        w = self.obs_weights
        res[0:2 * len(u):2] = (u - self.obs_uv[:, 0]) * w
        res[1:2 * len(u):2] = (v - self.obs_uv[:, 1]) * w
        # negatív mélység büntetése: nagy residual, hogy ne "forduljon át" pont
        bad = z <= 0.0
        if bad.any():
            res[0:2 * len(u):2][bad] = 1e3
            res[1:2 * len(u):2][bad] = 1e3
        o = 2 * len(u)
        for (pi, Ct, sigma) in self.pos_priors:
            R = Rs[pi]
            C = -R.T @ tvecs[pi]
            res[o:o + 3] = (C - Ct) / sigma
            o += 3
        return res

    # ------------------------------------------------------------ sparsity

    def sparsity(self):
        n_obs = len(self.obs_pidx)
        n_prior = len(self.pos_priors)
        n_pose = 6 * len(self.free_photos)
        n_pt = 3 * len(self.free_pt_idx)
        n_intr = len(self.refine_intr) * self.n_cams
        rows_l = []
        cols_l = []

        def add_block(row_idx, col_base, width):
            """row_idx (M,) sorokhoz [col_base, col_base+width) oszlopok."""
            m = len(row_idx)
            if m == 0 or width == 0:
                return
            rr = np.repeat(row_idx, width)
            cc = (col_base[:, None] + np.arange(width)[None, :]).ravel()
            rows_l.append(rr)
            cols_l.append(cc)

        obs_rows = np.arange(n_obs)
        ph_slot_arr = -np.ones(self.n_photos, np.int64)
        for p, s in self._ph_slot.items():
            ph_slot_arr[p] = s
        for k in range(2):
            r = 2 * obs_rows + k
            # póz-blokk
            s = ph_slot_arr[self.obs_cidx]
            sel = s >= 0
            add_block(r[sel], 6 * s[sel], 6)
            # pont-blokk
            sl = self._pt_slot[self.obs_pidx]
            sel = sl >= 0
            add_block(r[sel], n_pose + 3 * sl[sel], 3)
            # intrinsics-blokk
            if n_intr:
                ni = len(self.refine_intr)
                cid = self.photo_cam_ids[self.obs_cidx]
                add_block(r, n_pose + n_pt + ni * cid, ni)
        # priorok
        for j, (pi, _, _) in enumerate(self.pos_priors):
            s = self._ph_slot.get(int(pi))
            if s is not None:
                base = 2 * n_obs + 3 * j
                add_block(np.array([base, base + 1, base + 2]),
                          np.full(3, 6 * s, np.int64), 6)
        rr = np.concatenate(rows_l) if rows_l else np.zeros(0, np.int64)
        cc = np.concatenate(cols_l) if cols_l else np.zeros(0, np.int64)
        A = coo_matrix((np.ones(len(rr), np.uint8), (rr, cc)),
                       shape=(2 * n_obs + 3 * n_prior, n_pose + n_pt + n_intr))
        return A.tocsr()

    # -------------------------------------------------------------- futtatás

    def solve(self, loss="soft_l1", f_scale=2.0, max_nfev=60, verbose=0, log=print):
        x0 = self.pack()
        if len(x0) == 0 or len(self.obs_pidx) == 0:
            return self
        res = least_squares(self.residuals, x0, jac_sparsity=self.sparsity(),
                            method="trf", tr_solver="lsmr", loss=loss,
                            f_scale=f_scale, max_nfev=max_nfev, verbose=verbose,
                            x_scale="jac", ftol=1e-6, xtol=1e-8)
        self.rvecs, self.tvecs, self.points, self.cam_params = self.unpack(res.x)
        r = res.fun[:2 * len(self.obs_pidx)]
        w = np.repeat(self.obs_weights, 2)
        r = r / np.where(w == 0, 1, w)
        err = np.sqrt(r[0::2] ** 2 + r[1::2] ** 2)
        log(f"  BA kész: RMS={np.sqrt((err**2).mean()):.2f}px "
            f"medián={np.median(err):.2f}px nfev={res.nfev}")
        return self

    def reproj_errors(self):
        r = self.residuals(self.pack())[:2 * len(self.obs_pidx)]
        w = np.repeat(self.obs_weights, 2)
        r = r / np.where(w == 0, 1, w)
        return np.sqrt(r[0::2] ** 2 + r[1::2] ** 2)
