"""Dense cloud: kalibrált sztereópárok + SGBM mélységtérképek, referenciaképenkénti
több-partneres fúzióval (medián + konzisztencia-szűrés).

Minden beállt képhez legfeljebb `n_partners` szomszédot választunk; a
párokból mélységtérkép készül a referenciakép pixelrácsán, majd pixelenkénti
mediánnal és egyezés-ellenőrzéssel áll össze a végső mélység. Ez kiszűri az
egy-egy párra jellemző SGBM-torzítást és a durva hibákat.

Minőségi szintek (feldolgozási felbontás a teljeshez képest):
  high = 1/2, medium = 1/4, low = 1/8
"""

import cv2
import numpy as np

from .io_ply import write_ply_points

DENSE_SCALE = {"high": 0.5, "medium": 0.25, "low": 0.125}


def _shared_track_counts(project):
    from collections import defaultdict
    obs = project.observations
    pt_photos = defaultdict(list)
    for p_i, c_i in zip(obs[:, 0].astype(int), obs[:, 1].astype(int)):
        pt_photos[p_i].append(c_i)
    counts = defaultdict(int)
    for photos in pt_photos.values():
        photos = sorted(set(photos))
        for i in range(len(photos)):
            for j in range(i + 1, len(photos)):
                counts[(photos[i], photos[j])] += 1
    return counts


def _depth_range(project, photo_idx):
    obs = project.observations
    sel = obs[:, 1].astype(int) == photo_idx
    pts = project.points[obs[sel, 0].astype(int)]
    photo = project.photos[photo_idx]
    z = ((pts - photo.C[None, :]) @ photo.R.T)[:, 2]
    z = z[z > 0]
    if len(z) < 10:
        return None
    return float(np.percentile(z, 2) * 0.7), float(np.percentile(z, 98) * 1.4)


def select_stereo_partners(project, n_partners=3, log=print):
    """Minden beállt képhez a legjobb partnerek. Visszatér: [(i, [j...], zrange)]."""
    counts = _shared_track_counts(project)
    aligned = [i for i, p in enumerate(project.photos) if p.aligned]
    out = []
    for i in aligned:
        rng = _depth_range(project, i)
        if rng is None:
            continue
        mean_z = 0.5 * (rng[0] / 0.7 + rng[1] / 1.4)
        scored = []
        for j in aligned:
            if j == i:
                continue
            c = counts.get((min(i, j), max(i, j)), 0)
            if c < 50:
                continue
            B = np.linalg.norm(project.photos[i].C - project.photos[j].C)
            ratio = B / mean_z
            if not (0.05 <= ratio <= 0.7):
                continue
            score = c * np.exp(-((ratio - 0.25) / 0.15) ** 2 / 2)
            scored.append((score, j))
        scored.sort(reverse=True)
        if scored:
            out.append((i, [j for _, j in scored[:n_partners]], rng))
    log(f"  {len(out)} referenciakép, képenként max {n_partners} partner")
    return out


def _load_scaled(path, scale):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Nem olvasható: {path}")
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    return img


def _pair_depth_in_ref(project, i, j, img_i, img_j, zrange, scale):
    """Mélységtérkép az i kép (méretezett) pixelrácsán a j partnerrel.

    Visszatér: (h, w) float tömb, NaN ahol nincs adat.
    """
    pi, pj = project.photos[i], project.photos[j]
    cam_i = project.cameras[pi.cam_id]
    cam_j = project.cameras[pj.cam_id]
    h, w = img_i.shape[:2]

    def K_scaled(cam):
        # pixelközép-konvenció: u_s = s*u_f + (s-1)/2
        K = cam.K().copy()
        K[0, 0] *= scale; K[1, 1] *= scale
        K[0, 2] = scale * K[0, 2] + (scale - 1) / 2.0
        K[1, 2] = scale * K[1, 2] + (scale - 1) / 2.0
        return K

    K1, K2 = K_scaled(cam_i), K_scaled(cam_j)
    d1, d2 = cam_i.dist_coeffs(), cam_j.dist_coeffs()
    R = pj.R @ pi.R.T
    T = pj.R @ (pi.C - pj.C)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, d1, K2, d2, (w, h), R, T.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
    map1x, map1y = cv2.initUndistortRectifyMap(K1, d1, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, d2, R2, P2, (w, h), cv2.CV_32FC1)
    g_i = cv2.remap(cv2.cvtColor(img_i, cv2.COLOR_BGR2GRAY), map1x, map1y,
                    cv2.INTER_LINEAR)
    g_j = cv2.remap(cv2.cvtColor(img_j, cv2.COLOR_BGR2GRAY), map2x, map2y,
                    cv2.INTER_LINEAR)

    f_rect = abs(P1[0, 0])
    horizontal = abs(P2[0, 3]) >= abs(P2[1, 3])
    T_signed = (-P2[0, 3] if horizontal else -P2[1, 3]) / f_rect
    if not horizontal:
        g_i, g_j = cv2.transpose(g_i), cv2.transpose(g_j)

    da = f_rect * T_signed / zrange[1]
    db = f_rect * T_signed / zrange[0]
    d_lo, d_hi = min(da, db), max(da, db)
    min_disp = int(np.floor(d_lo / 16) * 16)
    num_disp = int(np.ceil((d_hi - min_disp) / 16) * 16)
    num_disp = max(16, min(num_disp, 16 * 48))

    bs = 7
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp, numDisparities=num_disp, blockSize=bs,
        P1=8 * bs * bs, P2=32 * bs * bs,
        uniquenessRatio=5, speckleWindowSize=150, speckleRange=2,
        disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_HH)
    disp = sgbm.compute(g_i, g_j).astype(np.float32) / 16.0
    if not horizontal:
        disp = cv2.transpose(disp)

    valid = disp > min_disp + 0.5
    if not valid.any():
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        z_rect = f_rect * T_signed / disp
    z_rect[~valid] = np.nan
    ok = (z_rect > zrange[0]) & (z_rect < zrange[1])
    z_rect[~ok] = np.nan

    # az i kép eredeti pixelrácsának rektifikált koordinátái
    # (undistortPoints R=R1, P=P1 beállítással pont ezt a leképezést adja)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).reshape(-1, 1, 2)
    rect_uv = cv2.undistortPoints(grid, K1, d1, R=R1, P=P1).reshape(h, w, 2)
    xr = rect_uv[..., 0].astype(np.float32)
    yr = rect_uv[..., 1].astype(np.float32)
    z_in_ref = cv2.remap(z_rect, xr, yr, cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    # rektifikált keretbeli pont: [ (xr-c1x)/f, (yr-c1y)/f, 1 ] * z_rect;
    # kamera keretben p_cam = R1^T p_rect -> z_cam = R1[:,2] @ p_rect
    c1x, c1y = P1[0, 2], P1[1, 2]
    col3 = R1[:, 2]
    z_cam = (col3[0] * (xr - c1x) / f_rect
             + col3[1] * (yr - c1y) / f_rect
             + col3[2]) * z_in_ref
    return z_cam.astype(np.float32)


def _unproject_ref(project, i, depth, scale, img_i):
    """Az i kép (méretezett) rácsán adott kamera-z mélységből world-pontok."""
    photo = project.photos[i]
    cam = project.cameras[photo.cam_id]
    h, w = depth.shape
    valid = np.isfinite(depth)
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return None
    z = depth[ys, xs].astype(np.float64)
    # pixel -> normalizált (torzításmentes) irány
    pts = np.stack([xs, ys], axis=1).astype(np.float64).reshape(-1, 1, 2)
    K = cam.K().copy()
    K[0, 0] *= scale; K[1, 1] *= scale
    K[0, 2] = scale * K[0, 2] + (scale - 1) / 2.0
    K[1, 2] = scale * K[1, 2] + (scale - 1) / 2.0
    und = cv2.undistortPoints(pts, K, cam.dist_coeffs()).reshape(-1, 2)
    dirs = np.hstack([und, np.ones((len(und), 1))])
    pts_cam = dirs * z[:, None]
    pts_world = pts_cam @ photo.R + photo.C[None, :]
    colors = img_i[ys, xs][:, ::-1]
    return pts_world, colors


def build_dense_cloud(project, quality="medium", n_partners=3,
                      agree_tol=0.01, out_path=None, log=print,
                      progress=None):
    """Dense cloud építése több-partneres fúzióval.

    agree_tol: relatív mélység-egyezési tűrés a partnerek között (1% ~ z*0.01)
    Visszatér: (points Nx3 world, colors Nx3 uint8 RGB)
    """
    if project.points is None or not len(project.points):
        raise RuntimeError("Előbb futtasd az Align photos lépést.")
    scale = DENSE_SCALE.get(quality, 0.25)
    refs = select_stereo_partners(project, n_partners=n_partners, log=log)
    if not refs:
        raise RuntimeError("Nincs használható sztereópár.")
    all_pts, all_col = [], []
    for k, (i, partners, zrange) in enumerate(refs):
        img_i = _load_scaled(project.photos[i].path, scale)
        depths = []
        for j in partners:
            img_j = _load_scaled(project.photos[j].path, scale)
            d = _pair_depth_in_ref(project, i, j, img_i, img_j, zrange, scale)
            if d is not None:
                depths.append(d)
        if not depths:
            log(f"  [{k+1}/{len(refs)}] {project.photos[i].name}: nincs mélység")
            continue
        D = np.stack(depths)                      # (k, h, w)
        cnt = np.isfinite(D).sum(0)
        med = np.nanmedian(D, axis=0)
        need = 2 if len(depths) >= 2 else 1
        agree = (np.abs(D - med[None]) < (agree_tol * med)[None]).sum(0)
        fused = np.where((cnt >= need) & (agree >= need), med, np.nan)
        r = _unproject_ref(project, i, fused.astype(np.float32), scale, img_i)
        if r is None:
            log(f"  [{k+1}/{len(refs)}] {project.photos[i].name}: üres")
            continue
        pts, col = r
        all_pts.append(pts)
        all_col.append(col)
        log(f"  [{k+1}/{len(refs)}] {project.photos[i].name} "
            f"({len(depths)} partner): {len(pts):,} pont")
        if progress:
            progress((k + 1) / len(refs))
    if not all_pts:
        raise RuntimeError("A sűrű illesztés nem adott pontot.")
    pts = np.vstack(all_pts)
    col = np.vstack(all_col)
    log(f"Dense cloud: {len(pts):,} pont összesen")
    if out_path:
        write_ply_points(out_path, pts, col)
        log(f"Mentve: {out_path}")
    return pts, col
