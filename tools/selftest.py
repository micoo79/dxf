#!/usr/bin/env python3
"""PyShape önteszt: szintetikus adathalmazon végigfuttatja a teljes
munkafolyamatot és ellenőrzi a pontosságot az ismert valósághoz képest.

Használat:
  python tools/selftest.py [munkamappa]

Kb. 3-6 perc. A végén PASS/FAIL összegzést ír.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402


def main():
    work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="pyshape_selftest_")
    ds = os.path.join(work, "dataset")
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")

    print(f"Munkamappa: {work}")
    print("1) Szintetikus adathalmaz generálása (12 kép)...")
    subprocess.run([sys.executable, os.path.join(HERE, "make_synthetic_dataset.py"),
                    ds, "--images", "12"], check=True,
                   stdout=subprocess.DEVNULL)

    from pyshape.project import Project
    from pyshape.loader import add_photos
    from pyshape.sfm import align_photos
    from pyshape.georef import georeference_from_gps
    from pyshape.gcp import load_gcp_file, predict_marker_images, optimize_with_gcps
    from pyshape.dense import build_dense_cloud
    from pyshape.meshing import build_dsm, build_mesh, save_mesh_ply, save_dsm_geotiff
    from pyshape.ortho import build_orthophoto

    gt = json.load(open(os.path.join(ds, "ground_truth.json")))
    K = np.array(gt["K"])
    gt_pose = {im["name"]: (np.array(im["R_wc"]), np.array(im["C"]))
               for im in gt["images"]}

    print("2) Képek betöltése (EXIF GPS -> EOV)...")
    prj = Project(os.path.join(work, "teszt.pshape"))
    added = add_photos(prj, [os.path.join(ds, "images")], log=lambda *a: None)
    check("12 kép betöltve EOV-pozícióval", len(added) == 12 and
          all(p.gps_eov for p in prj.photos))

    print("3) Align photos...")
    t0 = time.time()
    stats = align_photos(prj, quality="high", log=lambda *a: None)
    check("minden kép beállt", stats["aligned"] == 12,
          f"({stats['aligned']}/12, {time.time()-t0:.0f}s)")

    print("4) Georeferálás GPS-ből...")
    georeference_from_gps(prj, log=lambda *a: None)
    check("georeferálva", prj.georeferenced)

    print("5) GCP-k betöltése és 'kézi' bemérése (szimulált kattintás)...")
    load_gcp_file(prj, os.path.join(ds, "gcps.csv"), log=lambda *a: None)
    rng = np.random.default_rng(1)
    for m in prj.markers:
        cand = predict_marker_images(prj, m)
        used = 0
        X = np.array(m.eov)
        for photo, _, _, _ in cand:
            R, C = gt_pose[photo.name]
            xc = R @ (X - C)
            if xc[2] <= 0:
                continue
            uv = K @ (xc / xc[2])
            if 20 <= uv[0] < gt["width"] - 20 and 20 <= uv[1] < gt["height"] - 20:
                m.pixels[photo.name] = (float(uv[0] + rng.normal(0, 0.3)),
                                        float(uv[1] + rng.normal(0, 0.3)))
                used += 1
            if used >= 4:
                break
    n_meas = sum(1 for m in prj.markers if len(m.pixels) >= 2)
    check("GCP-k bemérve >=2 képen", n_meas >= 6, f"({n_meas} db)")

    print("6) Kamerák optimalizálása GCP-kkel...")
    rows = optimize_with_gcps(prj, log=lambda *a: None)
    gcp_rms = float(np.sqrt(np.mean([r["err_3d_m"] ** 2 for r in rows])))
    check("GCP RMS < 10 cm", gcp_rms < 0.10, f"({gcp_rms*100:.1f} cm)")
    prj.save()

    print("7) Dense cloud...")
    pts, col = build_dense_cloud(prj, quality="medium",
                                 out_path=os.path.join(prj.path, "dense.ply"),
                                 log=lambda *a: None)
    check("dense cloud > 50e pont", len(pts) > 50_000, f"({len(pts):,})")

    # pontosság a GT domborzathoz
    ter = gt["terrain"]
    hf = np.load(os.path.join(ds, "gt_heightfield.npy"))

    def dz_of(eov_pts):
        u = (eov_pts[:, 0] - ter["e0"]) / ter["esize"] * (hf.shape[1] - 1)
        v = (eov_pts[:, 1] - ter["n0"]) / ter["nsize"] * (hf.shape[0] - 1)
        ins = (u > 2) & (u < hf.shape[1] - 3) & (v > 2) & (v < hf.shape[0] - 3)
        u, v = u[ins], v[ins]
        iu, iv = u.astype(int), v.astype(int)
        fu, fv = u - iu, v - iv
        g = (hf[iv, iu] * (1 - fu) * (1 - fv) + hf[iv, iu + 1] * fu * (1 - fv)
             + hf[iv + 1, iu] * (1 - fu) * fv + hf[iv + 1, iu + 1] * fu * fv)
        return eov_pts[ins, 2] - (g + ter["base_h"])

    dz = dz_of(prj.world_to_eov(pts))
    check("dense magassági medián < 10 cm", abs(np.median(dz)) < 0.10,
          f"({np.median(dz)*100:+.1f} cm, MAE {np.abs(dz).mean()*100:.1f} cm)")

    print("8) DSM + mesh...")
    dsm = build_dsm(pts, col, log=lambda *a: None)
    save_dsm_geotiff(os.path.join(prj.path, "dsm.tif"), dsm, prj.origin,
                     log=lambda *a: None)
    verts, faces, vcol = build_mesh(dsm, log=lambda *a: None)
    save_mesh_ply(os.path.join(prj.path, "mesh.ply"), verts, faces, vcol,
                  origin=prj.origin)
    check("mesh épült", len(faces) > 10_000, f"({len(faces):,} háromszög)")

    print("9) Orthofotó + GeoTIFF...")
    ortho_path = os.path.join(prj.path, "ortho.tif")
    build_orthophoto(prj, dsm, out_path=ortho_path, log=lambda *a: None)

    import rasterio
    with rasterio.open(ortho_path) as src:
        check("GeoTIFF CRS = EPSG:23700", str(src.crs) == "EPSG:23700")
        img = src.read()
        inv = ~src.transform
        found = 0
        for g in gt["gcps"]:
            cx, cy = inv * (g["E"], g["N"])
            cx, cy = int(round(cx)), int(round(cy))
            if not (10 <= cx < src.width - 10 and 10 <= cy < src.height - 10):
                continue
            patch = img[:3, cy - 10:cy + 11, cx - 10:cx + 11]
            if patch.max() > 200:
                found += 1
        check("GCP-jelek a helyükön az orthofotón", found >= 6,
              f"({found}/{len(gt['gcps'])})")

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    print("\n==========================")
    print(f"Összesen: {len(checks)} ellenőrzés, {n_fail} hiba")
    print("EREDMÉNY:", "PASS" if n_fail == 0 else "FAIL")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
