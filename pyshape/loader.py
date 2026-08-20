"""Képek betöltése a projektbe: EXIF GPS -> EOV, kameramodell-csoportosítás,
opcionális referenciafájl (kép;E;N;H) beolvasása.
"""

import csv
import io
import os

import numpy as np

from .exif import read_exif, estimate_focal_px
from .geodesy import wgs84_to_eov
from .project import CameraModel, Photo

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _sniff_rows(path):
    """Rugalmas CSV-olvasó: ';' ',' tab vagy szóköz elválasztóval, # komment."""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for sep in (";", ",", "\t"):
                if sep in line:
                    parts = [p.strip() for p in line.split(sep)]
                    break
            else:
                parts = line.split()
            rows.append(parts)
    return rows


def load_reference_csv(path):
    """Kamerapozíció-referencia: kép_név;E;N;H sorok. dict: name -> (E,N,H)."""
    out = {}
    for parts in _sniff_rows(path):
        if len(parts) < 4:
            continue
        try:
            e, n, h = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue  # fejlécsor
        out[parts[0]] = (e, n, h)
    return out


def add_photos(project, paths, reference_csv=None, geoid_offset=0.0, log=print,
               progress=None):
    """Képek hozzáadása a projekthez.

    - EXIF-ből kiolvassa a GPS-t és WGS84-ből EOV-ba (EPSG:23700) transzformálja.
    - Ha `reference_csv` meg van adva, az felülírja az EXIF-pozíciót.
    - A képeket (méret + fókusz) szerint kameracsoportokba sorolja.
    Visszatér: a hozzáadott Photo objektumok listája.
    """
    ref = load_reference_csv(reference_csv) if reference_csv else {}

    # mappák kibontása
    files = []
    for p in paths:
        if os.path.isdir(p):
            for fn in sorted(os.listdir(p)):
                if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                    files.append(os.path.join(p, fn))
        else:
            files.append(p)

    if not files:
        log("! Nem található képfájl a megadott helyen "
            "(támogatott: jpg, jpeg, png, tif, tiff, bmp).")
        return []
    log(f"{len(files)} képfájl feldolgozása...")

    added = []
    existing = {p.name for p in project.photos}
    for file_i, fp in enumerate(files):
        if progress:
            progress((file_i + 1) / len(files))
        name = os.path.basename(fp)
        if name in existing:
            log(f"  kihagyva (már betöltve): {name}")
            continue
        try:
            info = read_exif(fp)
        except Exception as e:
            log(f"  ! nem olvasható, kihagyva: {name} ({e})")
            continue
        photo = Photo(fp, name)
        photo.exif = {k: info[k] for k in ("f35", "focal_mm", "make", "model")}
        w, h = info["width"], info["height"]

        # kameracsoport keresése/létrehozása
        f_px = estimate_focal_px(info, w) or 1.2 * max(w, h)
        cam_id = None
        for i, c in enumerate(project.cameras):
            if c.width == w and c.height == h and abs(c.f - f_px) < 1e-6:
                cam_id = i
                break
        if cam_id is None:
            project.cameras.append(CameraModel(w, h, f_px))
            cam_id = len(project.cameras) - 1
        photo.cam_id = cam_id

        # pozíció: referenciafájl > EXIF
        if name in ref:
            photo.gps_eov = ref[name]
        elif info["lat"] is not None and info["lon"] is not None:
            photo.gps_wgs = (info["lon"], info["lat"], info["alt"] or 0.0)
            photo.gps_eov = wgs84_to_eov(info["lon"], info["lat"],
                                         info["alt"] or 0.0, geoid_offset)
        project.photos.append(photo)
        added.append(photo)
        pos = ""
        if photo.gps_eov:
            pos = f"  EOV=({photo.gps_eov[0]:.2f}, {photo.gps_eov[1]:.2f}, {photo.gps_eov[2]:.2f})"
        log(f"  + {name}  {w}x{h}  f={f_px:.0f}px{pos}")

    # world-origó felvétele az első GPS-es képek átlagából
    if project.origin is None:
        pts = np.array([p.gps_eov for p in project.photos if p.gps_eov])
        if len(pts):
            project.origin = (float(round(pts[:, 0].mean())),
                              float(round(pts[:, 1].mean())),
                              0.0)
            log(f"Lokális origó (EOV): {project.origin}")
    return added
