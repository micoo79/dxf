"""DSM (digitális felszínmodell) és 2.5D mesh a dense cloudból.

A DSM egy szabályos EOV-rács, cellánként robusztus (medián) magassággal,
lyukkitöltéssel és tüskeszűréssel. A mesh a DSM-rács 2.5D háromszögelése
(TIN), csúcsszínekkel — PLY-ba exportálható.
"""

import numpy as np
import cv2
from scipy import ndimage

from .io_ply import write_ply_mesh


class DSM:
    """Szabályos rács world-keretben. z[i,j]: i=sor (N csökken), j=oszlop (E nő)."""

    def __init__(self, e_min, n_max, gsd, z, colors=None, valid=None):
        self.e_min = e_min      # world (EOV - origin) koordináta!
        self.n_max = n_max
        self.gsd = gsd
        self.z = z              # (rows, cols) float32, NaN = nincs adat
        self.colors = colors    # (rows, cols, 3) uint8 vagy None
        self.valid = valid if valid is not None else np.isfinite(z)

    @property
    def shape(self):
        return self.z.shape

    def cell_centers(self):
        rows, cols = self.z.shape
        e = self.e_min + (np.arange(cols) + 0.5) * self.gsd
        n = self.n_max - (np.arange(rows) + 0.5) * self.gsd
        return e, n

    def sample(self, e, n):
        """Bilineáris magasság-mintavétel world (E,N) koordinátákon."""
        rows, cols = self.z.shape
        x = (np.asarray(e) - self.e_min) / self.gsd - 0.5
        y = (self.n_max - np.asarray(n)) / self.gsd - 0.5
        x = np.clip(x, 0, cols - 1 - 1e-9)
        y = np.clip(y, 0, rows - 1 - 1e-9)
        ix = x.astype(np.int64); iy = y.astype(np.int64)
        fx = x - ix; fy = y - iy
        z = self.z
        return (z[iy, ix] * (1 - fx) * (1 - fy) + z[iy, ix + 1] * fx * (1 - fy)
                + z[iy + 1, ix] * (1 - fx) * fy + z[iy + 1, ix + 1] * fx * fy)


def build_dsm(points, colors=None, gsd=None, fill_holes=True,
              median_filter=True, log=print):
    """DSM építése pontfelhőből (world-keretben).

    gsd: cellaméret méterben; None esetén automatikus a pontsűrűségből.
    """
    pts = np.asarray(points)
    e_min, n_min = pts[:, 0].min(), pts[:, 1].min()
    e_max, n_max = pts[:, 0].max(), pts[:, 1].max()
    area = max((e_max - e_min) * (n_max - n_min), 1e-9)
    if gsd is None:
        # cellánként átlagosan ~4 pont
        gsd = float(np.sqrt(4.0 * area / len(pts)))
        gsd = max(gsd, 0.01)
    cols = max(int(np.ceil((e_max - e_min) / gsd)), 1)
    rows = max(int(np.ceil((n_max - n_min) / gsd)), 1)
    if rows * cols > 100_000_000:
        raise RuntimeError(f"Túl nagy DSM-rács ({rows}x{cols}) — növeld a GSD-t.")
    log(f"  DSM-rács: {rows} x {cols} cella, GSD={gsd:.3f} m")

    j = np.clip(((pts[:, 0] - e_min) / gsd).astype(np.int64), 0, cols - 1)
    i = np.clip(((n_max - pts[:, 1]) / gsd).astype(np.int64), 0, rows - 1)
    flat = i * cols + j

    # cellánkénti medián magasság (rendezéssel, vektorosan)
    order = np.argsort(flat, kind="stable")
    fs = flat[order]
    zs = pts[order, 2]
    uniq, start = np.unique(fs, return_index=True)
    end = np.r_[start[1:], len(fs)]
    z_grid = np.full(rows * cols, np.nan, np.float32)
    med = np.empty(len(uniq))
    for k in range(len(uniq)):     # cellánkénti kis szeletek — gyors
        med[k] = np.median(zs[start[k]:end[k]])
    z_grid[uniq] = med
    z_grid = z_grid.reshape(rows, cols)

    col_grid = None
    if colors is not None:
        cs = colors[order].astype(np.float64)
        sums = np.add.reduceat(cs, start, axis=0)
        cnt = (end - start)[:, None]
        col_grid = np.zeros((rows * cols, 3), np.uint8)
        col_grid[uniq] = np.clip(sums / cnt, 0, 255).astype(np.uint8)
        col_grid = col_grid.reshape(rows, cols, 3)

    valid = np.isfinite(z_grid)
    log(f"  kitöltöttség: {valid.mean()*100:.1f}%")

    if median_filter:
        # tüskeszűrés: ahol a cella nagyon eltér a környezet mediánjától
        zf = z_grid.copy()
        zf[~valid] = np.nanmedian(z_grid)
        med9 = cv2.medianBlur(zf.astype(np.float32), 5)
        spikes = valid & (np.abs(z_grid - med9) > 6 * np.nanstd(z_grid - med9) + 1e-6)
        if spikes.any():
            z_grid[spikes] = med9[spikes]
            log(f"  {spikes.sum()} tüske simítva")

    if fill_holes and (~valid).any():
        # legközelebbi érvényes cella értéke + enyhe simítás a kitöltött részen
        ind = ndimage.distance_transform_edt(~valid, return_distances=False,
                                             return_indices=True)
        filled = z_grid[tuple(ind)]
        smooth = cv2.blur(filled.astype(np.float32), (5, 5))
        z_out = z_grid.copy()
        z_out[~valid] = smooth[~valid]
        z_grid = z_out
        if col_grid is not None:
            col_grid = np.where(valid[..., None], col_grid,
                                col_grid[tuple(ind)])
        log(f"  {int((~valid).sum())} cella lyukkitöltéssel pótolva")

    return DSM(e_min, n_max, gsd, z_grid.astype(np.float32), col_grid, valid)


def build_mesh(dsm, decimate=1, log=print):
    """2.5D TIN mesh a DSM-ből. Visszatér (verts Nx3 world, faces Mx3, colors)."""
    z = dsm.z[::decimate, ::decimate]
    valid = np.isfinite(z)
    rows, cols = z.shape
    gsd = dsm.gsd * decimate
    e = dsm.e_min + (np.arange(cols) + 0.5) * gsd
    n = dsm.n_max - (np.arange(rows) + 0.5) * gsd
    E, N = np.meshgrid(e, n)
    vid = -np.ones((rows, cols), np.int64)
    vy, vx = np.nonzero(valid)
    vid[vy, vx] = np.arange(len(vy))
    verts = np.stack([E[vy, vx], N[vy, vx], z[vy, vx]], axis=1)
    colors = None
    if dsm.colors is not None:
        colors = dsm.colors[::decimate, ::decimate][vy, vx]

    # két háromszög minden olyan cellanégyeshez, ahol mind a 4 csúcs érvényes
    v00 = vid[:-1, :-1]; v01 = vid[:-1, 1:]
    v10 = vid[1:, :-1]; v11 = vid[1:, 1:]
    ok = (v00 >= 0) & (v01 >= 0) & (v10 >= 0) & (v11 >= 0)
    a = v00[ok]; b = v01[ok]; c = v10[ok]; d = v11[ok]
    faces = np.concatenate([np.stack([a, c, b], axis=1),
                            np.stack([b, c, d], axis=1)])
    log(f"  mesh: {len(verts):,} csúcs, {len(faces):,} háromszög")
    return verts, faces, colors


def save_mesh_ply(path, verts, faces, colors=None, origin=None):
    """Mesh mentése PLY-ba. origin megadásakor EOV-ba tolja a csúcsokat."""
    v = verts.copy()
    if origin is not None:
        v = v + np.asarray(origin)[None, :]
    write_ply_mesh(path, v, faces, colors)


def save_dsm_geotiff(path, dsm, origin, crs="EPSG:23700", log=print):
    """DSM exportálása GeoTIFF-be (EOV)."""
    import rasterio
    from rasterio.transform import from_origin
    e0, n0, h0 = origin
    transform = from_origin(dsm.e_min + e0, dsm.n_max + n0, dsm.gsd, dsm.gsd)
    z = dsm.z + h0
    with rasterio.open(
            path, "w", driver="GTiff", height=z.shape[0], width=z.shape[1],
            count=1, dtype="float32", crs=crs, transform=transform,
            nodata=-9999.0, compress="deflate") as dst:
        out = np.where(np.isfinite(z), z, -9999.0).astype(np.float32)
        dst.write(out, 1)
    log(f"  DSM GeoTIFF mentve: {path}")
