"""Georeferálás: hasonlósági (7 paraméteres Helmert-) transzformáció az SfM-keret
és az EOV-alapú lokális world-keret között.

Két forrásból dolgozik:
  - kamera-GPS (EXIF/referenciafájl) — durva georeferálás align után
  - GCP-k — pontos georeferálás (a gcp modul hívja)

Síkban fekvő kameraelrendezésnél (tipikus drónrepülés) az Umeyama-megoldás
kétértelmű (a terep a kamerák "fölé" is kerülhet); ezt a kötőpontok
elhelyezkedésével (a terepnek a kamerák alatt kell lennie) oldjuk fel.
"""

import cv2
import numpy as np


def umeyama(src, dst, weights=None):
    """Súlyozott hasonlósági transzformáció: dst ~ s * R @ src + t."""
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    w = np.ones(len(src)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    mu_s = (src * w[:, None]).sum(0)
    mu_d = (dst * w[:, None]).sum(0)
    ds = src - mu_s; dd = dst - mu_d
    S = (dd * w[:, None]).T @ ds
    U, D, Vt = np.linalg.svd(S)
    sgn = np.eye(3)
    sgn[2, 2] = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    R = U @ sgn @ Vt
    var = (w[:, None] * ds ** 2).sum()
    s = np.trace(np.diag(D) @ sgn) / max(var, 1e-12)
    t = mu_d - s * R @ mu_s
    return s, R, t


def _plane_basis(pts):
    """Ponthalmaz legjobban illeszkedő síkjának (normál, síkbeli tengely, centroid)."""
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c)
    a1 = vt[0]      # legnagyobb szórású irány (síkban)
    n = vt[2]       # normál
    return n, a1, c


def similarity_terrain_aware(src, dst, terrain_pts=None, weights=None, log=print):
    """Umeyama + síkdegeneráció feloldása.

    terrain_pts: pontok az src keretében, amelyeknek a transzformáció után a
    kamerák (dst magasságai) ALATT kell lenniük (légi felvételezés feltevés).
    """
    s, R, t = umeyama(src, dst, weights)
    if terrain_pts is None or len(terrain_pts) == 0:
        return s, R, t
    cam_z = (s * (R @ src.T).T + t)[:, 2].mean()
    ter_z = np.median((s * (R @ np.asarray(terrain_pts).T).T + t)[:, 2])
    if ter_z < cam_z:
        return s, R, t
    # rossz oldal: 180°-os forgatás a forrás-sík egyik síkbeli tengelye körül,
    # majd újra Umeyama — ez a "másik" valódi forgatás-minimum.
    log("  georef: sík-kétértelműség feloldása (terep a kamerák fölé került)")
    n, a1, c = _plane_basis(src)
    Rf, _ = cv2.Rodrigues(a1 * np.pi)
    src2 = (src - c) @ Rf.T + c
    s2, R2, t2 = umeyama(src2, dst, weights)
    R_c = R2 @ Rf
    t_c = t2 + s2 * R2 @ (c - Rf @ c)
    cam_z = (s2 * (R_c @ src.T).T + t_c)[:, 2].mean()
    ter_z = np.median((s2 * (R_c @ np.asarray(terrain_pts).T).T + t_c)[:, 2])
    if ter_z > cam_z:
        log("  ! figyelem: a terep így is a kamerák fölött van — ellenőrizd a GPS-t")
    return s2, R_c, t_c


def apply_similarity(project, s, R, t):
    """A teljes rekonstrukció áttranszformálása: X' = s R X + t."""
    for p in project.photos:
        if p.aligned:
            p.C = s * R @ p.C + t
            p.R = p.R @ R.T
    if project.points is not None and len(project.points):
        project.points = (s * (R @ project.points.T).T + t)


def georeference_from_gps(project, log=print):
    """Durva georeferálás a kamera-GPS pozíciók alapján (world = EOV - origin)."""
    src, dst = [], []
    for p in project.photos:
        if p.aligned and p.gps_eov is not None:
            src.append(p.C)
            dst.append(project.eov_to_world(np.array(p.gps_eov))[0])
    if len(src) < 3:
        raise RuntimeError("Legalább 3 GPS-pozíciós, beállt kép kell a georeferáláshoz.")
    src = np.array(src); dst = np.array(dst)
    s, R, t = similarity_terrain_aware(src, dst, terrain_pts=project.points, log=log)
    apply_similarity(project, s, R, t)
    project.georeferenced = True
    res = dst - np.array([p.C for p in project.photos
                          if p.aligned and p.gps_eov is not None])
    rms = np.sqrt((res ** 2).sum(1).mean())
    log(f"Georeferálás (GPS): skála={s:.4f}, kamera-RMS={rms:.2f} m")
    return {"scale": s, "camera_rms_m": float(rms)}
