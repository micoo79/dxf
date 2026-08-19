"""GCP-k (földi illesztőpontok) kezelése.

  - GCP-fájl betöltése (név;E;N;H — ',' ';' tab vagy szóköz elválasztóval)
  - annak megjóslása, mely képeken látszik egy GCP (kézi azonosításhoz)
  - GCP-mérések triangulálása
  - "Optimize cameras": GCP-kényszeres kötegelt kiegyenlítés, amely a teljes
    blokkot az EOV-keretbe húzza (Metashape 'optimize' megfelelője)
"""

import numpy as np

from .ba import BAProblem, rt_from_pose, pose_from_rt
from .georef import similarity_terrain_aware, apply_similarity, umeyama
from .loader import _sniff_rows
from .project import Marker
from .sfm import _normalize, _triangulate_nviews


def load_gcp_file(project, path, log=print):
    """GCP-k beolvasása fájlból a projektbe. Visszatér: új Marker-lista."""
    added = []
    existing = {m.name for m in project.markers}
    for parts in _sniff_rows(path):
        if len(parts) < 4:
            continue
        try:
            e, n, h = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue  # fejléc
        name = parts[0]
        if name in existing:
            log(f"  kihagyva (már létezik): {name}")
            continue
        m = Marker(name, (e, n, h))
        project.markers.append(m)
        added.append(m)
        log(f"  + {name}: E={e:.3f} N={n:.3f} H={h:.3f}")
    if project.origin is None and added:
        pts = np.array([m.eov for m in added])
        project.origin = (float(round(pts[:, 0].mean())),
                          float(round(pts[:, 1].mean())), 0.0)
        log(f"Lokális origó (EOV): {project.origin}")
    return added


def predict_marker_images(project, marker, margin=0.03):
    """Mely (beállt) képeken látszódhat a GCP?

    Csak georeferált projektben értelmes. Visszatér: [(photo, u, v, közép-táv)]
    a képközépponthoz mért relatív távolság szerint rendezve.
    """
    if not project.georeferenced:
        return []
    Xw = project.eov_to_world(np.array(marker.eov))[0]
    out = []
    for photo in project.photos:
        if not photo.aligned:
            continue
        cam = project.cameras[photo.cam_id]
        uv, z = photo.project(Xw[None, :], cam)
        u, v = uv[0]
        if z[0] <= 0:
            continue
        mw = margin * cam.width
        mh = margin * cam.height
        if -mw <= u < cam.width + mw and -mh <= v < cam.height + mh:
            du = (u - cam.width / 2) / cam.width
            dv = (v - cam.height / 2) / cam.height
            out.append((photo, float(u), float(v), float(np.hypot(du, dv))))
    out.sort(key=lambda r: r[3])
    return out


def triangulate_marker(project, marker):
    """GCP kézi méréseinek triangulálása a jelenlegi world-keretben."""
    Ps, xs = [], []
    for name, uv in marker.pixels.items():
        photo = project.photo_by_name(name)
        if photo is None or not photo.aligned:
            continue
        cam = project.cameras[photo.cam_id]
        t = -photo.R @ photo.C
        Ps.append(np.hstack([photo.R, t.reshape(3, 1)]))
        xs.append(_normalize(np.array(uv, float), cam)[0])
    if len(Ps) < 2:
        return None
    return _triangulate_nviews(Ps, xs)


def marker_errors(project, log=print):
    """GCP-hibariport: 3D eltérés (m) és képi reprojekciós hiba (px)."""
    rows = []
    for m in project.markers:
        if not m.enabled or len(m.pixels) < 2:
            continue
        X = triangulate_marker(project, m)
        if X is None:
            continue
        eov_est = project.world_to_eov(X)[0]
        d = eov_est - np.array(m.eov)
        # reprojekciós hiba
        Xw = project.eov_to_world(np.array(m.eov))[0]
        errs = []
        for name, uv in m.pixels.items():
            photo = project.photo_by_name(name)
            if photo is None or not photo.aligned:
                continue
            cam = project.cameras[photo.cam_id]
            p_uv, _ = photo.project(Xw[None, :], cam)
            errs.append(float(np.hypot(p_uv[0][0] - uv[0], p_uv[0][1] - uv[1])))
        rows.append({"name": m.name, "n_img": len(m.pixels),
                     "dE": float(d[0]), "dN": float(d[1]), "dH": float(d[2]),
                     "err_3d_m": float(np.linalg.norm(d)),
                     "err_px": float(np.mean(errs)) if errs else None})
    if rows:
        tot = np.sqrt(np.mean([r["err_3d_m"] ** 2 for r in rows]))
        for r in rows:
            log(f"  {r['name']}: dE={r['dE']*100:+.1f}cm dN={r['dN']*100:+.1f}cm "
                f"dH={r['dH']*100:+.1f}cm | {r['err_px']:.2f}px ({r['n_img']} kép)")
        log(f"  GCP összesített RMS: {tot*100:.1f} cm")
    return rows


def optimize_with_gcps(project, gps_sigma=3.0, marker_weight=4.0, log=print):
    """Kamerák optimalizálása GCP-kényszerekkel (Metashape 'Optimize Cameras').

    1. hasonlósági transzformáció frissítése a triangulált GCP-k -> EOV alapján
    2. teljes BA: GCP-pontok rögzítve EOV-ban, mérések felsúlyozva,
       kamera-GPS gyenge priorként, intrinsics finomítás
    """
    measured = [m for m in project.markers if m.enabled and len(m.pixels) >= 2]
    if len(measured) < 3:
        raise RuntimeError("Legalább 3, két-két képen bemért GCP szükséges.")

    # ---- 1) hasonlósági frissítés (GCP-k erősen, kamerák gyengén súlyozva)
    src, dst, wgt = [], [], []
    for m in measured:
        X = triangulate_marker(project, m)
        if X is None:
            continue
        src.append(X)
        dst.append(project.eov_to_world(np.array(m.eov))[0])
        wgt.append(100.0)
    n_gcp_fit = len(src)
    for p in project.photos:
        if p.aligned and p.gps_eov is not None:
            src.append(p.C)
            dst.append(project.eov_to_world(np.array(p.gps_eov))[0])
            wgt.append(1.0)
    if n_gcp_fit >= 3:
        cam_pts = np.array([p.C for p in project.photos if p.aligned])
        s, R, t = similarity_terrain_aware(np.array(src), np.array(dst),
                                           terrain_pts=project.points,
                                           cam_pts=cam_pts,
                                           weights=np.array(wgt), log=log)
        apply_similarity(project, s, R, t)
        project.georeferenced = True
        log(f"  hasonlósági frissítés: skála={s:.6f}")

    # ---- 2) GCP-kényszeres BA
    photos = project.photos
    aligned_idx = [i for i, p in enumerate(photos) if p.aligned]
    lidx = {g: l for l, g in enumerate(aligned_idx)}
    rvecs = np.zeros((len(aligned_idx), 3))
    tvecs = np.zeros((len(aligned_idx), 3))
    for g, l in lidx.items():
        rvecs[l], tvecs[l] = rt_from_pose(photos[g].R, photos[g].C)

    pts = project.points.copy()
    obs = project.observations
    obs_p = obs[:, 0].astype(np.int64)
    obs_c_global = obs[:, 1].astype(np.int64)
    keep = np.array([g in lidx for g in obs_c_global])
    obs_p = obs_p[keep]
    obs_c = np.array([lidx[g] for g in obs_c_global[keep]])
    obs_uv = obs[keep, 2:4]
    weights = np.ones(len(obs_p))

    # GCP-pontok hozzáfűzése rögzített koordinátával
    fixed = np.zeros(len(pts) + len(measured), bool)
    extra_pts, extra_obs = [], []
    for k, m in enumerate(measured):
        Xw = project.eov_to_world(np.array(m.eov))[0]
        extra_pts.append(Xw)
        fixed[len(pts) + k] = True
        for name, uv in m.pixels.items():
            photo = project.photo_by_name(name)
            if photo is None or not photo.aligned:
                continue
            g = photos.index(photo)
            extra_obs.append((len(pts) + k, lidx[g], uv[0], uv[1]))
    pts = np.vstack([pts, np.array(extra_pts)])
    if extra_obs:
        eo = np.array(extra_obs)
        obs_p = np.concatenate([obs_p, eo[:, 0].astype(np.int64)])
        obs_c = np.concatenate([obs_c, eo[:, 1].astype(np.int64)])
        obs_uv = np.vstack([obs_uv, eo[:, 2:4]])
        weights = np.concatenate([weights, np.full(len(eo), marker_weight)])

    priors = []
    for g, l in lidx.items():
        if photos[g].gps_eov is not None:
            Ct = project.eov_to_world(np.array(photos[g].gps_eov))[0]
            priors.append((l, Ct, gps_sigma))

    cam_params = [c.to_dict() for c in project.cameras]
    for cp in cam_params:
        cp.pop("width"); cp.pop("height")
    prob = BAProblem(pts, obs_p, obs_c, obs_uv, rvecs, tvecs,
                     [photos[g].cam_id for g in aligned_idx],
                     cam_params, None,
                     refine_intrinsics=("f", "cx", "cy", "k1", "k2"),
                     fixed_points=fixed, obs_weights=weights,
                     pos_priors=priors)
    log("  GCP-kényszeres kötegelt kiegyenlítés...")
    prob.solve(max_nfev=120, log=log)

    # visszaírás
    for g, l in lidx.items():
        R, C = pose_from_rt(prob.rvecs[l], prob.tvecs[l])
        photos[g].R = R
        photos[g].C = C
    project.points = prob.points[:len(project.points)]
    for ci, cp in enumerate(prob.cam_params):
        c = project.cameras[ci]
        for k, v in cp.items():
            setattr(c, k, float(v))
    log("  kamera az optimalizálás után: " + ", ".join(
        f"f={c.f:.1f}px k1={c.k1:.2e} k2={c.k2:.2e}" for c in project.cameras))
    return marker_errors(project, log=log)
