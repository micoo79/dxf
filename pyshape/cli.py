"""Parancssori felület — a teljes munkafolyamat GUI nélkül is futtatható.

Példa (teljes pipeline):
  python -m pyshape new munka.pshape --images fotok/
  python -m pyshape align munka.pshape --quality high
  python -m pyshape georef munka.pshape
  python -m pyshape gcp-load munka.pshape gcps.csv
  python -m pyshape gcp-measure munka.pshape meresek.csv
  python -m pyshape optimize munka.pshape
  python -m pyshape dense munka.pshape --quality medium
  python -m pyshape mesh munka.pshape
  python -m pyshape ortho munka.pshape -o ortho.tif

A GCP-k kézi azonosításához a grafikus felület való:
  python -m pyshape gui munka.pshape
"""

import argparse
import os
import sys

import numpy as np


def _load(path):
    from .project import Project
    if not os.path.exists(os.path.join(path, "project.json")):
        sys.exit(f"Nincs ilyen projekt: {path}")
    return Project.load(path)


def cmd_new(args):
    from .project import Project
    from .loader import add_photos
    prj = Project(args.project)
    added = add_photos(prj, args.images, reference_csv=args.reference,
                       geoid_offset=args.geoid_offset)
    if not added:
        sys.exit("Nem található kép.")
    prj.save()
    print(f"Projekt létrehozva: {args.project} ({len(added)} kép)")


def cmd_add_photos(args):
    from .loader import add_photos
    prj = _load(args.project)
    add_photos(prj, args.images, reference_csv=args.reference,
               geoid_offset=args.geoid_offset)
    prj.save()


def cmd_align(args):
    from .sfm import align_photos
    from .georef import georeference_from_gps
    prj = _load(args.project)
    align_photos(prj, quality=args.quality)
    # ha van GPS, rögtön georeferálunk is
    if sum(1 for p in prj.photos if p.aligned and p.gps_eov) >= 3:
        georeference_from_gps(prj)
    prj.save()


def cmd_georef(args):
    from .georef import georeference_from_gps
    prj = _load(args.project)
    georeference_from_gps(prj)
    prj.save()


def cmd_gcp_load(args):
    from .gcp import load_gcp_file
    prj = _load(args.project)
    load_gcp_file(prj, args.file)
    prj.save()


def cmd_gcp_list(args):
    from .gcp import predict_marker_images
    prj = _load(args.project)
    for m in prj.markers:
        print(f"{m.name}: EOV=({m.eov[0]:.3f}, {m.eov[1]:.3f}, {m.eov[2]:.3f}) "
              f"— {len(m.pixels)} mérés")
        for photo, u, v, _ in predict_marker_images(prj, m):
            mark = " *" if photo.name in m.pixels else ""
            print(f"    látható: {photo.name} ~({u:.0f},{v:.0f}){mark}")


def cmd_gcp_measure(args):
    """Mérések betöltése fájlból: gcp_nev;kep_nev;u;v soronként."""
    from .loader import _sniff_rows
    prj = _load(args.project)
    markers = {m.name: m for m in prj.markers}
    n = 0
    for parts in _sniff_rows(args.file):
        if len(parts) < 4 or parts[0] not in markers:
            continue
        try:
            u, v = float(parts[2]), float(parts[3])
        except ValueError:
            continue
        markers[parts[0]].pixels[parts[1]] = (u, v)
        n += 1
    prj.save()
    print(f"{n} mérés betöltve.")


def cmd_optimize(args):
    from .gcp import optimize_with_gcps
    prj = _load(args.project)
    optimize_with_gcps(prj, gps_sigma=args.gps_sigma)
    prj.save()


def cmd_dense(args):
    from .dense import build_dense_cloud
    prj = _load(args.project)
    out = os.path.join(args.project, "dense.ply")
    build_dense_cloud(prj, quality=args.quality, out_path=out)
    prj.save()


def cmd_mesh(args):
    from .io_ply import read_ply_points
    from .meshing import build_dsm, build_mesh, save_mesh_ply, save_dsm_geotiff
    prj = _load(args.project)
    dense_path = os.path.join(args.project, "dense.ply")
    if not os.path.exists(dense_path):
        sys.exit("Előbb futtasd a dense lépést.")
    pts, col = read_ply_points(dense_path)
    dsm = build_dsm(pts, col, gsd=args.gsd)
    save_dsm_geotiff(os.path.join(args.project, "dsm.tif"), dsm, prj.origin)
    verts, faces, vcol = build_mesh(dsm)
    save_mesh_ply(os.path.join(args.project, "mesh.ply"), verts, faces, vcol,
                  origin=prj.origin)
    print("Mesh és DSM elmentve a projektmappába.")


def load_dsm_from_project(prj):
    """A projektmappában lévő dsm.tif visszatöltése DSM objektummá."""
    import rasterio
    from .meshing import DSM
    path = os.path.join(prj.path, "dsm.tif")
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        z = src.read(1).astype(np.float32)
        z[z == src.nodata] = np.nan
        t = src.transform
        gsd = t.a
        e_min = t.c - prj.origin[0]
        n_max = t.f - prj.origin[1]
    z = z - prj.origin[2]
    return DSM(e_min, n_max, gsd, z)


def cmd_ortho(args):
    from .ortho import build_orthophoto
    prj = _load(args.project)
    dsm = load_dsm_from_project(prj)
    if dsm is None:
        sys.exit("Előbb futtasd a mesh lépést (az készíti a DSM-et).")
    out = args.output or os.path.join(args.project, "ortho.tif")
    build_orthophoto(prj, dsm, gsd=args.gsd, blend=args.blend, out_path=out)


def cmd_status(args):
    prj = _load(args.project)
    print(f"Projekt: {args.project}")
    print(f"  képek: {len(prj.photos)} (beállt: {len(prj.aligned_photos())})")
    print(f"  kötőpontok: {0 if prj.points is None else len(prj.points):,}")
    print(f"  georeferált: {'igen' if prj.georeferenced else 'nem'} "
          f"(origó: {prj.origin})")
    print(f"  GCP-k: {len(prj.markers)} "
          f"(bemérve: {sum(1 for m in prj.markers if len(m.pixels) >= 2)})")
    for name in ("dense.ply", "mesh.ply", "dsm.tif", "ortho.tif"):
        p = os.path.join(args.project, name)
        if os.path.exists(p):
            print(f"  {name}: {os.path.getsize(p)/1e6:.1f} MB")


def cmd_gui(args):
    from .gui.app import run_gui
    run_gui(args.project)


def build_parser():
    ap = argparse.ArgumentParser(
        prog="pyshape",
        description="PyShape — fotogrammetriai munkafolyamat (Metashape-szerű)")
    sub = ap.add_subparsers(dest="cmd")

    def add(name, fn, help):
        p = sub.add_parser(name, help=help)
        p.set_defaults(fn=fn)
        return p

    p = add("gui", cmd_gui, "grafikus felület indítása")
    p.add_argument("project", nargs="?", default=None)

    p = add("new", cmd_new, "új projekt képekből")
    p.add_argument("project")
    p.add_argument("--images", nargs="+", required=True,
                   help="képfájlok vagy mappák")
    p.add_argument("--reference", default=None,
                   help="kamerapozíció-referencia CSV (kep;E;N;H)")
    p.add_argument("--geoid-offset", type=float, default=0.0,
                   help="geoidunduláció (m): EOV magasság = GPS-magasság - offset")

    p = add("add-photos", cmd_add_photos, "képek hozzáadása")
    p.add_argument("project")
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--reference", default=None)
    p.add_argument("--geoid-offset", type=float, default=0.0)

    p = add("align", cmd_align, "képek beállítása (SfM + BA)")
    p.add_argument("project")
    p.add_argument("--quality", default="high",
                   choices=["highest", "high", "medium", "low"])

    p = add("georef", cmd_georef, "georeferálás kamera-GPS alapján")
    p.add_argument("project")

    p = add("gcp-load", cmd_gcp_load, "GCP-k betöltése fájlból (nev;E;N;H)")
    p.add_argument("project")
    p.add_argument("file")

    p = add("gcp-list", cmd_gcp_list, "GCP-k és láthatóságuk listázása")
    p.add_argument("project")

    p = add("gcp-measure", cmd_gcp_measure,
            "képi mérések betöltése (gcp;kep;u;v)")
    p.add_argument("project")
    p.add_argument("file")

    p = add("optimize", cmd_optimize, "kamerák optimalizálása GCP-kkel")
    p.add_argument("project")
    p.add_argument("--gps-sigma", type=float, default=3.0)

    p = add("dense", cmd_dense, "dense cloud építése")
    p.add_argument("project")
    p.add_argument("--quality", default="medium",
                   choices=["high", "medium", "low"])

    p = add("mesh", cmd_mesh, "DSM és mesh építése")
    p.add_argument("project")
    p.add_argument("--gsd", type=float, default=None, help="DSM cellaméret (m)")

    p = add("ortho", cmd_ortho, "orthofotó + GeoTIFF export")
    p.add_argument("project")
    p.add_argument("--gsd", type=float, default=None, help="ortho felbontás (m)")
    p.add_argument("--blend", default="weighted", choices=["weighted", "best"])
    p.add_argument("-o", "--output", default=None)

    p = add("status", cmd_status, "projekt állapota")
    p.add_argument("project")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "cmd", None):
        # alapértelmezés: GUI
        args = ap.parse_args(["gui"])
    args.fn(args)


if __name__ == "__main__":
    main()
