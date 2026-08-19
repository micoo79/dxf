"""Egyszerű bináris PLY írás/olvasás (pontfelhő és mesh)."""

import numpy as np


def write_ply_points(path, pts, colors=None):
    n = len(pts)
    with open(path, "wb") as f:
        hdr = ["ply", "format binary_little_endian 1.0",
               f"element vertex {n}",
               "property float x", "property float y", "property float z"]
        if colors is not None:
            hdr += ["property uchar red", "property uchar green",
                    "property uchar blue"]
        hdr += ["end_header", ""]
        f.write("\n".join(hdr).encode())
        if colors is not None:
            rec = np.zeros(n, dtype=[("xyz", np.float32, 3),
                                     ("rgb", np.uint8, 3)])
            rec["xyz"] = pts.astype(np.float32)
            rec["rgb"] = colors.astype(np.uint8)
        else:
            rec = np.zeros(n, dtype=[("xyz", np.float32, 3)])
            rec["xyz"] = pts.astype(np.float32)
        rec.tofile(f)


def write_ply_mesh(path, verts, faces, colors=None):
    n, m = len(verts), len(faces)
    with open(path, "wb") as f:
        hdr = ["ply", "format binary_little_endian 1.0",
               f"element vertex {n}",
               "property float x", "property float y", "property float z"]
        if colors is not None:
            hdr += ["property uchar red", "property uchar green",
                    "property uchar blue"]
        hdr += [f"element face {m}", "property list uchar int vertex_indices",
                "end_header", ""]
        f.write("\n".join(hdr).encode())
        if colors is not None:
            rec = np.zeros(n, dtype=[("xyz", np.float32, 3),
                                     ("rgb", np.uint8, 3)])
            rec["xyz"] = verts.astype(np.float32)
            rec["rgb"] = colors.astype(np.uint8)
        else:
            rec = np.zeros(n, dtype=[("xyz", np.float32, 3)])
            rec["xyz"] = verts.astype(np.float32)
        rec.tofile(f)
        frec = np.zeros(m, dtype=[("n", np.uint8), ("idx", np.int32, 3)])
        frec["n"] = 3
        frec["idx"] = faces.astype(np.int32)
        frec.tofile(f)


def read_ply_points(path):
    """Csak a saját írónk által készített formátumot olvassa vissza."""
    with open(path, "rb") as f:
        line = f.readline().strip()
        assert line == b"ply"
        n = 0
        has_color = False
        while True:
            line = f.readline().strip()
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            elif line.startswith(b"property uchar red"):
                has_color = True
            elif line == b"end_header":
                break
        if has_color:
            rec = np.fromfile(f, dtype=[("xyz", np.float32, 3),
                                        ("rgb", np.uint8, 3)], count=n)
            return rec["xyz"].astype(np.float64), rec["rgb"].copy()
        rec = np.fromfile(f, dtype=[("xyz", np.float32, 3)], count=n)
        return rec["xyz"].astype(np.float64), None
