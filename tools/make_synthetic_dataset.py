#!/usr/bin/env python3
"""Szintetikus légifényképezési adathalmaz generálása a PyShape teszteléséhez.

Domborzatmodell (heightfield) + procedurális földfelszín-textúra fölött
pinhole kamerákkal rendereli le a képeket, EXIF GPS-adatokkal (WGS84,
EOV-ból visszaszámolva), és kiírja:

  images/IMG_####.jpg   - renderelt légifotók EXIF GPS-szel
  gcps.csv              - földi illesztőpontok (név;E;N;H) EOV-ban
  reference.csv         - kamerapozíciók (név;E;N;H) EOV-ban
  ground_truth.json     - pontos kamerapózok/intrinsics a pontosságméréshez

Használat:
  python make_synthetic_dataset.py <kimeneti_mappa> [--images N] [--width W]
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

try:
    import piexif
except ImportError:
    piexif = None

from pyproj import Transformer

EOV_TO_WGS = Transformer.from_crs("EPSG:23700", "EPSG:4326", always_xy=True)

# ---------------------------------------------------------------- terrain ---

def make_heightfield(rng, nx, ny, amplitude):
    """Sima, több oktávos véletlen domborzat."""
    z = np.zeros((ny, nx), np.float64)
    for octave, scale in ((6, 1.0), (12, 0.5), (24, 0.25)):
        small = rng.standard_normal((octave, octave))
        z += scale * cv2.resize(small, (nx, ny), interpolation=cv2.INTER_CUBIC)
    z -= z.mean()
    z *= amplitude / max(1e-9, np.abs(z).max())
    return z


def make_texture(rng, tw, th):
    """Procedurális földfelszín-textúra: talaj/növényzet foltok, utak, vonalak."""
    base = np.zeros((th, tw, 3), np.float32)
    # nagyléptékű szín-variáció (zöld-barna mezők)
    for _ in range(3):
        n = rng.standard_normal((rng.integers(6, 14), rng.integers(6, 14), 3)).astype(np.float32)
        base += cv2.resize(n, (tw, th), interpolation=cv2.INTER_CUBIC) * 18.0
    ground = np.array([95.0, 110.0, 80.0], np.float32)  # BGR-ben zöldes-barnás
    img = base + ground
    # közepes léptékű foltok
    n = rng.standard_normal((th // 8, tw // 8, 3)).astype(np.float32)
    img += cv2.resize(n, (tw, th), interpolation=cv2.INTER_LINEAR) * 10.0
    # finom szemcse — enélkül a SIFT-nek nincs mibe kapaszkodnia
    img += rng.standard_normal((th, tw, 3)).astype(np.float32) * 12.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    # "utak": világos csíkok
    for _ in range(6):
        p1 = (int(rng.integers(0, tw)), int(rng.integers(0, th)))
        p2 = (int(rng.integers(0, tw)), int(rng.integers(0, th)))
        cv2.line(img, p1, p2, (170, 170, 165), thickness=int(rng.integers(6, 20)))
    # sötét parcella-határok
    for _ in range(25):
        p1 = (int(rng.integers(0, tw)), int(rng.integers(0, th)))
        ang = rng.uniform(0, 2 * math.pi)
        ln = rng.uniform(60, 400)
        p2 = (int(p1[0] + ln * math.cos(ang)), int(p1[1] + ln * math.sin(ang)))
        cv2.line(img, p1, p2, (60, 65, 55), thickness=int(rng.integers(2, 5)))
    # világos foltok (kavics, épületalap)
    for _ in range(40):
        c = (int(rng.integers(0, tw)), int(rng.integers(0, th)))
        cv2.circle(img, c, int(rng.integers(4, 30)),
                   tuple(int(v) for v in rng.integers(120, 220, 3)), -1)
    # újabb szemcseréteg, hogy a rajzolt elemek is textúráltak legyenek
    noise = rng.standard_normal((th, tw, 1)).astype(np.float32) * 7.0
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def draw_gcp_marker(tex, px, py, radius_px):
    """Fekete-fehér céltábla-jel a textúrán (px, py textúra-pixel)."""
    r = int(radius_px)
    cv2.circle(tex, (px, py), r, (255, 255, 255), -1)
    cv2.circle(tex, (px, py), r, (0, 0, 0), max(1, r // 5))
    t = max(1, r // 4)
    cv2.line(tex, (px - r, py), (px + r, py), (0, 0, 0), t)
    cv2.line(tex, (px, py - r), (px, py + r), (0, 0, 0), t)


# --------------------------------------------------------------- rendering --

def rot_matrix(yaw, pitch, roll):
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll); world->cam a transzponáltja lesz."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


class Terrain:
    def __init__(self, e0, n0, esize, nsize, hf, base_h):
        self.e0, self.n0 = e0, n0
        self.esize, self.nsize = esize, nsize
        self.hf = hf
        self.base_h = base_h
        self.ny, self.nx = hf.shape

    def height(self, e, n):
        """Bilineáris magasság-mintavétel (vektorosan)."""
        u = (e - self.e0) / self.esize * (self.nx - 1)
        v = (n - self.n0) / self.nsize * (self.ny - 1)
        u = np.clip(u, 0, self.nx - 1 - 1e-6)
        v = np.clip(v, 0, self.ny - 1 - 1e-6)
        iu = u.astype(np.int64); iv = v.astype(np.int64)
        fu = u - iu; fv = v - iv
        h = (self.hf[iv, iu] * (1 - fu) * (1 - fv)
             + self.hf[iv, iu + 1] * fu * (1 - fv)
             + self.hf[iv + 1, iu] * (1 - fu) * fv
             + self.hf[iv + 1, iu + 1] * fu * fv)
        return h + self.base_h


def render_image(terrain, tex, cam_center, R_wc, K, w, h, rng):
    """Sugármenetes renderelés: minden pixelhez heightfield-metszés + textúra."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    # kamera-koordinátás irányvektorok (z előre = lefelé néző tengely irányában)
    d_cam = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)], axis=-1)
    d_world = d_cam.reshape(-1, 3) @ R_wc  # R_wc: world->cam, ezért d @ R = R^T @ d
    d_world /= np.linalg.norm(d_world, axis=1, keepdims=True)

    zmax = terrain.base_h + terrain.hf.max()
    zmin = terrain.base_h + terrain.hf.min()
    dz = d_world[:, 2]
    # csak lefelé tartó sugarak érdekesek; a többi kap "eget" (nem fordul elő nadírnál)
    valid = dz < -1e-6
    t0 = (zmax - cam_center[2]) / np.where(valid, dz, -1.0)
    t1 = (zmin - cam_center[2]) / np.where(valid, dz, -1.0)
    t0 = np.maximum(t0, 0.0)

    n_steps = 96
    t = t0.copy()
    step = (t1 - t0) / n_steps
    hit_t = t1.copy()
    hit_found = np.zeros(d_world.shape[0], bool)
    prev_t = t0.copy()
    for _ in range(n_steps + 1):
        p = cam_center[None, :] + d_world * t[:, None]
        ground = terrain.height(p[:, 0], p[:, 1])
        below = (p[:, 2] <= ground) & valid & ~hit_found
        hit_t[below] = t[below]
        hit_found |= below
        prev_t = np.where(~hit_found, t, prev_t)
        t = t + step
    # bisection finomítás a [hit_t - step, hit_t] intervallumon
    lo = np.maximum(hit_t - step, t0)
    hi = hit_t
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        p = cam_center[None, :] + d_world * mid[:, None]
        ground = terrain.height(p[:, 0], p[:, 1])
        below = p[:, 2] <= ground
        hi = np.where(below, mid, hi)
        lo = np.where(below, lo, mid)
    tfin = 0.5 * (lo + hi)
    p = cam_center[None, :] + d_world * tfin[:, None]

    # textúra-mintavétel
    th, tw = tex.shape[:2]
    u = (p[:, 0] - terrain.e0) / terrain.esize * (tw - 1)
    v = (1.0 - (p[:, 1] - terrain.n0) / terrain.nsize) * (th - 1)  # N felfelé nő, kép lefelé
    inside = (u >= 0) & (u <= tw - 1) & (v >= 0) & (v <= th - 1) & hit_found
    u = np.clip(u, 0, tw - 1 - 1e-6); v = np.clip(v, 0, th - 1 - 1e-6)
    iu = u.astype(np.int64); iv = v.astype(np.int64)
    fu = (u - iu)[:, None]; fv = (v - iv)[:, None]
    texf = tex.astype(np.float32)
    c = (texf[iv, iu] * (1 - fu) * (1 - fv)
         + texf[iv, iu + 1] * fu * (1 - fv)
         + texf[iv + 1, iu] * (1 - fu) * fv
         + texf[iv + 1, iu + 1] * fu * fv)
    c[~inside] = 24.0  # területen kívül sötét
    img = c.reshape(h, w, 3)
    img += rng.standard_normal(img.shape).astype(np.float32) * 2.0  # szenzorzaj
    return np.clip(img, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------- exif ---

def deg_to_dms_rational(deg):
    d = int(deg)
    m_f = (deg - d) * 60
    m = int(m_f)
    s = round((m_f - m) * 60 * 10000)
    return ((d, 1), (m, 1), (s, 10000))


def write_jpeg_with_exif(path, img, e, n, h_m, f35mm):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    data = buf.tobytes()
    if piexif is None:
        with open(path, "wb") as f:
            f.write(data)
        return
    lon, lat = EOV_TO_WGS.transform(e, n)
    gps = {
        piexif.GPSIFD.GPSLatitudeRef: b"N" if lat >= 0 else b"S",
        piexif.GPSIFD.GPSLatitude: deg_to_dms_rational(abs(lat)),
        piexif.GPSIFD.GPSLongitudeRef: b"E" if lon >= 0 else b"W",
        piexif.GPSIFD.GPSLongitude: deg_to_dms_rational(abs(lon)),
        piexif.GPSIFD.GPSAltitudeRef: 0,
        piexif.GPSIFD.GPSAltitude: (int(round(h_m * 100)), 100),
    }
    exif_ifd = {
        piexif.ExifIFD.FocalLengthIn35mmFilm: int(round(f35mm)),
        piexif.ExifIFD.FocalLength: (int(round(f35mm / 36 * 132 * 100)), 100),
    }
    zeroth = {piexif.ImageIFD.Make: b"PyShape", piexif.ImageIFD.Model: b"SynthCam"}
    exif_bytes = piexif.dump({"0th": zeroth, "Exif": exif_ifd, "GPS": gps})
    piexif.insert(exif_bytes, data, path)


# -------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--images", type=int, default=12)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = args.out
    os.makedirs(os.path.join(out, "images"), exist_ok=True)

    # terület EOV-ban
    e0, n0 = 650000.0, 240000.0
    esize, nsize = 200.0, 150.0
    base_h = 120.0
    terrain = Terrain(e0, n0, esize, nsize,
                      make_heightfield(rng, 256, 192, amplitude=5.0), base_h)

    # textúra ~5 cm/px
    tw, th = 4000, 3000
    tex = make_texture(rng, tw, th)

    # GCP-k (jól szétszórva, a szélektől beljebb)
    gcp_rel = [(0.18, 0.2), (0.5, 0.15), (0.82, 0.22),
               (0.2, 0.8), (0.55, 0.85), (0.8, 0.75), (0.5, 0.5)]
    gcps = []
    for i, (fx_, fy_) in enumerate(gcp_rel):
        e = e0 + fx_ * esize
        n = n0 + fy_ * nsize
        hgt = float(terrain.height(np.array([e]), np.array([n]))[0])
        px = int(fx_ * (tw - 1))
        py = int((1 - fy_) * (th - 1))
        draw_gcp_marker(tex, px, py, radius_px=14)
        gcps.append((f"GCP{i+1}", e, n, hgt))

    # kamera-elrendezés
    w, h = args.width, args.height
    f_px = 1200.0 * (w / 1440.0)
    K = np.array([[f_px, 0, (w - 1) / 2.0], [0, f_px, (h - 1) / 2.0], [0, 0, 1.0]])
    f35 = f_px / w * 36.0

    fly_h = 110.0  # repülési magasság a terep átlagszintje felett
    ncols = 4
    nrows = max(1, args.images // ncols)
    e_positions = np.linspace(e0 + 40, e0 + esize - 40, ncols)
    n_positions = np.linspace(n0 + 30, n0 + nsize - 30, nrows)

    images_meta = []
    ref_rows = []
    idx = 0
    for r_i, n_c in enumerate(n_positions):
        cols = e_positions if r_i % 2 == 0 else e_positions[::-1]
        for e_c in cols:
            idx += 1
            name = f"IMG_{idx:04d}.jpg"
            e_c_ = e_c + rng.uniform(-2, 2)
            n_c_ = n_c + rng.uniform(-2, 2)
            z_c = base_h + fly_h + rng.uniform(-3, 3)
            yaw = rng.uniform(0, 2 * math.pi)
            pitch = math.radians(rng.uniform(-4, 4))
            roll = math.radians(rng.uniform(-4, 4))
            # kamera tengely lefelé: cam z tengely a -Z világirányba
            R_base = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])  # nadír
            R_wc = (rot_matrix(yaw, pitch, roll) @ R_base).T  # world->cam
            C = np.array([e_c_, n_c_, z_c])
            img = render_image(terrain, tex, C, R_wc, K, w, h, rng)
            # EXIF GPS zajjal (valósághű: RTK nélküli drón ~1-2 m)
            gps_e = e_c_ + rng.normal(0, 1.2)
            gps_n = n_c_ + rng.normal(0, 1.2)
            gps_h = z_c + rng.normal(0, 2.0)
            write_jpeg_with_exif(os.path.join(out, "images", name), img,
                                 gps_e, gps_n, gps_h, f35)
            ref_rows.append((name, gps_e, gps_n, gps_h))
            images_meta.append({
                "name": name,
                "C": C.tolist(),
                "R_wc": R_wc.tolist(),
            })
            print(f"  render {name}  C=({e_c_:.1f},{n_c_:.1f},{z_c:.1f})")
            if idx >= args.images:
                break
        if idx >= args.images:
            break

    with open(os.path.join(out, "gcps.csv"), "w") as f:
        f.write("# nev;E;N;H (EOV / EPSG:23700)\n")
        for name, e, n, hgt in gcps:
            f.write(f"{name};{e:.3f};{n:.3f};{hgt:.3f}\n")

    with open(os.path.join(out, "reference.csv"), "w") as f:
        f.write("# kep;E;N;H\n")
        for name, e, n, hgt in ref_rows:
            f.write(f"{name};{e:.3f};{n:.3f};{hgt:.3f}\n")

    gt = {
        "K": K.tolist(), "width": w, "height": h,
        "images": images_meta,
        "gcps": [{"name": g[0], "E": g[1], "N": g[2], "H": g[3]} for g in gcps],
        "terrain": {"e0": e0, "n0": n0, "esize": esize, "nsize": nsize,
                    "base_h": base_h},
    }
    with open(os.path.join(out, "ground_truth.json"), "w") as f:
        json.dump(gt, f, indent=1)
    # DSM ground truth mentése az összehasonlításhoz
    np.save(os.path.join(out, "gt_heightfield.npy"), terrain.hf)
    print(f"Kész: {idx} kép, {len(gcps)} GCP -> {out}")


if __name__ == "__main__":
    sys.exit(main())
