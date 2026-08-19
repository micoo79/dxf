"""Projektmodell: fényképek, kameramodell, kötőpontok, GCP-k, mentés/betöltés.

Koordinátakeretek:
  - "world": lokális méteres rendszer. Georeferálás után world = EOV - origin
    (origin = self.origin), Z = balti/ellipszoidi magasság a bemenettől függően.
  - Georeferálás előtt a world az SfM tetszőleges skálájú kerete.

A kamerapóz: x_cam = R @ (X_world - C), azaz R a world->cam forgatás, C a
vetítési középpont world-ben.
"""

import json
import os

import numpy as np


class CameraModel:
    """Közös belső tájékozás (intrinsics) egy kameracsoporthoz."""

    def __init__(self, width, height, f, cx=None, cy=None, k1=0.0, k2=0.0):
        self.width = int(width)
        self.height = int(height)
        self.f = float(f)
        self.cx = float(cx) if cx is not None else (width - 1) / 2.0
        self.cy = float(cy) if cy is not None else (height - 1) / 2.0
        self.k1 = float(k1)
        self.k2 = float(k2)

    def K(self):
        return np.array([[self.f, 0, self.cx],
                         [0, self.f, self.cy],
                         [0, 0, 1.0]])

    def dist_coeffs(self):
        # OpenCV sorrend: k1, k2, p1, p2
        return np.array([self.k1, self.k2, 0.0, 0.0])

    def to_dict(self):
        return {k: getattr(self, k) for k in
                ("width", "height", "f", "cx", "cy", "k1", "k2")}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class Photo:
    def __init__(self, path, name=None):
        self.path = path
        self.name = name or os.path.basename(path)
        self.cam_id = 0
        self.gps_eov = None      # (E,N,H) EOV-ban, EXIF-ből vagy referenciafájlból
        self.gps_wgs = None      # (lon, lat, alt) ha EXIF-ből jött
        self.R = None            # 3x3 world->cam
        self.C = None            # kameraközéppont world-ben
        self.exif = {}

    @property
    def aligned(self):
        return self.R is not None

    def project(self, X, cam):
        """world-pontok (Nx3) vetítése pixelbe. Visszatér (Nx2 uv, N mélység)."""
        X = np.atleast_2d(X)
        xc = (X - self.C[None, :]) @ self.R.T
        z = xc[:, 2]
        x = xc[:, 0] / np.where(z == 0, 1e-12, z)
        y = xc[:, 1] / np.where(z == 0, 1e-12, z)
        r2 = x * x + y * y
        d = 1.0 + cam.k1 * r2 + cam.k2 * r2 * r2
        u = cam.f * x * d + cam.cx
        v = cam.f * y * d + cam.cy
        return np.stack([u, v], axis=1), z

    def to_dict(self):
        d = {"path": self.path, "name": self.name, "cam_id": self.cam_id,
             "gps_eov": list(self.gps_eov) if self.gps_eov is not None else None,
             "gps_wgs": list(self.gps_wgs) if self.gps_wgs is not None else None,
             "R": self.R.tolist() if self.R is not None else None,
             "C": self.C.tolist() if self.C is not None else None}
        return d

    @classmethod
    def from_dict(cls, d):
        p = cls(d["path"], d["name"])
        p.cam_id = d.get("cam_id", 0)
        p.gps_eov = tuple(d["gps_eov"]) if d.get("gps_eov") else None
        p.gps_wgs = tuple(d["gps_wgs"]) if d.get("gps_wgs") else None
        if d.get("R") is not None:
            p.R = np.array(d["R"])
            p.C = np.array(d["C"])
        return p


class Marker:
    """Földi illesztőpont (GCP) EOV-koordinátával és kézi képi mérésekkel."""

    def __init__(self, name, eov):
        self.name = name
        self.eov = tuple(float(v) for v in eov)  # (E, N, H)
        self.pixels = {}     # photo_name -> (u, v) teljes felbontású pixel
        self.enabled = True

    def to_dict(self):
        return {"name": self.name, "eov": list(self.eov),
                "pixels": {k: list(v) for k, v in self.pixels.items()},
                "enabled": self.enabled}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["name"], d["eov"])
        m.pixels = {k: tuple(v) for k, v in d.get("pixels", {}).items()}
        m.enabled = d.get("enabled", True)
        return m


class Project:
    def __init__(self, path=None):
        self.path = path              # projektmappa (.pshape)
        self.photos = []              # list[Photo]
        self.cameras = []             # list[CameraModel]
        self.markers = []             # list[Marker]
        self.origin = None            # (E0, N0, H0) — world = EOV - origin
        self.georeferenced = False
        # ritka pontfelhő (align eredménye)
        self.points = None            # Nx3 world
        self.point_colors = None      # Nx3 uint8 (RGB)
        self.observations = None      # Mx4: (point_idx, photo_idx, u, v)
        self.log = []

    # ---------------------------------------------------------- segédek ----

    def photo_by_name(self, name):
        for p in self.photos:
            if p.name == name:
                return p
        return None

    def aligned_photos(self):
        return [p for p in self.photos if p.aligned]

    def world_to_eov(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return X + np.asarray(self.origin, float)[None, :]

    def eov_to_world(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return X - np.asarray(self.origin, float)[None, :]

    # ------------------------------------------------------ mentés/betöltés

    def save(self, path=None):
        path = path or self.path
        assert path, "Nincs projektútvonal"
        self.path = path
        os.makedirs(path, exist_ok=True)
        doc = {
            "version": 1,
            "origin": list(self.origin) if self.origin is not None else None,
            "georeferenced": self.georeferenced,
            "cameras": [c.to_dict() for c in self.cameras],
            "photos": [p.to_dict() for p in self.photos],
            "markers": [m.to_dict() for m in self.markers],
        }
        with open(os.path.join(path, "project.json"), "w") as f:
            json.dump(doc, f, indent=1)
        arrays = {}
        if self.points is not None:
            arrays["points"] = self.points
            arrays["point_colors"] = self.point_colors
            arrays["observations"] = self.observations
        if arrays:
            np.savez_compressed(os.path.join(path, "sparse.npz"), **arrays)

    @classmethod
    def load(cls, path):
        prj = cls(path)
        with open(os.path.join(path, "project.json")) as f:
            doc = json.load(f)
        prj.origin = tuple(doc["origin"]) if doc.get("origin") else None
        prj.georeferenced = doc.get("georeferenced", False)
        prj.cameras = [CameraModel.from_dict(d) for d in doc["cameras"]]
        prj.photos = [Photo.from_dict(d) for d in doc["photos"]]
        prj.markers = [Marker.from_dict(d) for d in doc["markers"]]
        sp = os.path.join(path, "sparse.npz")
        if os.path.exists(sp):
            z = np.load(sp)
            prj.points = z["points"] if "points" in z else None
            prj.point_colors = z["point_colors"] if "point_colors" in z else None
            prj.observations = z["observations"] if "observations" in z else None
        return prj
