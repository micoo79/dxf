"""Orthofotó (ortomozaik) készítése a DSM-re támaszkodva, GeoTIFF exporttal.

Minden ortho-cellához a DSM adja a magasságot; a cellát a beállt képekbe
vetítjük és a képközéphez közeli nézeteket súlyozva keverjük (a varratok a
képek közepe felé kerülnek, mint a Metashape mozaik módjában).
"""

import cv2
import numpy as np

from .geodesy import EOV_CRS


def estimate_image_gsd(project):
    """A képek átlagos terepi felbontása (m/pixel) a ritka pontok mélységéből."""
    obs = project.observations
    gsds = []
    for i, photo in enumerate(project.photos):
        if not photo.aligned:
            continue
        sel = obs[:, 1].astype(int) == i
        if not sel.any():
            continue
        pts = project.points[obs[sel, 0].astype(int)]
        z = ((pts - photo.C[None, :]) @ photo.R.T)[:, 2]
        z = z[z > 0]
        if len(z):
            cam = project.cameras[photo.cam_id]
            gsds.append(np.median(z) / cam.f)
    return float(np.median(gsds)) if gsds else None


def build_orthophoto(project, dsm, gsd=None, blend="weighted",
                     out_path=None, log=print, progress=None):
    """Orthofotó a DSM területére. Visszatér (rgb uint8 HxWx3, alpha bool, transform-adatok).

    gsd: ortho felbontás méterben (None = képek natív GSD-je)
    blend: 'weighted' (súlyozott keverés) vagy 'best' (legjobb nézet)
    """
    if gsd is None:
        gsd = estimate_image_gsd(project) or dsm.gsd
        gsd = max(gsd, dsm.gsd / 8)
    rows_d, cols_d = dsm.shape
    e_min = dsm.e_min
    n_max = dsm.n_max
    width = int(np.ceil(cols_d * dsm.gsd / gsd))
    height = int(np.ceil(rows_d * dsm.gsd / gsd))
    if width * height > 400_000_000:
        raise RuntimeError("Túl nagy orthofotó-rács — növeld a GSD-t.")
    log(f"  ortho-rács: {height} x {width} pixel, GSD={gsd:.3f} m")

    e = e_min + (np.arange(width) + 0.5) * gsd
    n = n_max - (np.arange(height) + 0.5) * gsd
    E, N = np.meshgrid(e, n)
    Z = dsm.sample(E.ravel(), N.ravel()).reshape(height, width)
    X = np.stack([E.ravel(), N.ravel(), Z.ravel()], axis=1)

    acc = np.zeros((height, width, 3), np.float32)
    wsum = np.zeros((height, width), np.float32)
    best_w = np.zeros((height, width), np.float32)

    photos = [p for p in project.photos if p.aligned]
    for k, photo in enumerate(photos):
        cam = project.cameras[photo.cam_id]
        uv, depth = photo.project(X, cam)
        u = uv[:, 0].reshape(height, width).astype(np.float32)
        v = uv[:, 1].reshape(height, width).astype(np.float32)
        d = depth.reshape(height, width)
        inside = ((u >= 0) & (u <= cam.width - 1) &
                  (v >= 0) & (v <= cam.height - 1) & (d > 0))
        if not inside.any():
            continue
        # súly: a képközéptől a szél felé csökken (0 a szélen, 1 középen)
        wu = np.minimum(u, cam.width - 1 - u) / (cam.width / 2)
        wv = np.minimum(v, cam.height - 1 - v) / (cam.height / 2)
        wgt = np.clip(np.minimum(wu, wv), 0, 1) ** 2
        wgt = np.where(inside, wgt, 0).astype(np.float32)
        from .imio import imread
        img = imread(photo.path, cv2.IMREAD_COLOR)
        if img is None:
            log(f"  ! nem olvasható kép, kihagyva: {photo.name}")
            continue
        col = cv2.remap(img, u, v, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        col = col[..., ::-1].astype(np.float32)  # BGR -> RGB
        if blend == "best":
            better = wgt > best_w
            acc[better] = col[better] * 1.0
            wsum[better] = 1.0
            best_w = np.maximum(best_w, wgt)
        else:
            acc += col * wgt[..., None]
            wsum += wgt
        log(f"  [{k+1}/{len(photos)}] {photo.name} lefedettség: "
            f"{inside.mean()*100:.0f}%")
        if progress:
            progress((k + 1) / len(photos))

    alpha = wsum > 1e-6
    rgb = np.zeros((height, width, 3), np.uint8)
    rgb[alpha] = np.clip(acc[alpha] / wsum[alpha][:, None], 0, 255).astype(np.uint8)
    meta = {"e_min": e_min, "n_max": n_max, "gsd": gsd}
    if out_path:
        save_ortho_geotiff(out_path, rgb, alpha, meta, project.origin, log=log)
    return rgb, alpha, meta


def save_ortho_geotiff(path, rgb, alpha, meta, origin, crs=EOV_CRS, log=print):
    """RGBA GeoTIFF export EOV-ban (EPSG:23700)."""
    import rasterio
    from rasterio.transform import from_origin
    e0, n0, _ = origin
    transform = from_origin(meta["e_min"] + e0, meta["n_max"] + n0,
                            meta["gsd"], meta["gsd"])
    h, w = rgb.shape[:2]
    with rasterio.open(
            path, "w", driver="GTiff", height=h, width=w, count=4,
            dtype="uint8", crs=crs, transform=transform,
            compress="deflate", photometric="RGB") as dst:
        for b in range(3):
            dst.write(rgb[..., b], b + 1)
        dst.write((alpha * 255).astype(np.uint8), 4)
        dst.colorinterp = [rasterio.enums.ColorInterp.red,
                           rasterio.enums.ColorInterp.green,
                           rasterio.enums.ColorInterp.blue,
                           rasterio.enums.ColorInterp.alpha]
    log(f"  Orthofotó GeoTIFF mentve: {path}")
