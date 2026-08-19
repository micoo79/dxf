"""Jellemzőpont-detektálás (SIFT) és képpárok illesztése.

A kulcspontok mindig TELJES felbontású pixelkoordinátában tárolódnak, a
detektálás viszont (minőségi beállítástól függően) kicsinyített képen fut.
"""

import os

import cv2
import numpy as np

QUALITY_SCALE = {"highest": 1.0, "high": 1.0, "medium": 0.5, "low": 0.25}
QUALITY_FEATURES = {"highest": 12000, "high": 8000, "medium": 6000, "low": 4000}


def load_image_scaled(path, max_dim=None, scale=None, gray=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Nem olvasható kép: {path}")
    s = 1.0
    if scale is not None:
        s = scale
    elif max_dim is not None and max(img.shape[:2]) > max_dim:
        s = max_dim / max(img.shape[:2])
    if s != 1.0:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img, s


def detect_features(project, quality="high", log=print):
    """SIFT kulcspontok minden képre. Visszatér: dict name -> (kp Nx2, desc NxD, colors Nx3)."""
    scale = QUALITY_SCALE.get(quality, 1.0)
    nfeat = QUALITY_FEATURES.get(quality, 8000)
    sift = cv2.SIFT_create(nfeatures=nfeat, contrastThreshold=0.02)
    feats = {}
    for i, photo in enumerate(project.photos):
        img, s = load_image_scaled(photo.path, scale=scale)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, desc = sift.detectAndCompute(gray, None)
        if desc is None or len(kps) == 0:
            feats[photo.name] = (np.zeros((0, 2)), np.zeros((0, 128), np.float32),
                                 np.zeros((0, 3), np.uint8))
            log(f"  ! nincs jellemzőpont: {photo.name}")
            continue
        pts = np.array([k.pt for k in kps], np.float64)
        xi = np.clip(pts[:, 0].astype(int), 0, img.shape[1] - 1)
        yi = np.clip(pts[:, 1].astype(int), 0, img.shape[0] - 1)
        colors = img[yi, xi][:, ::-1].copy()  # BGR -> RGB
        # vissza teljes felbontásra (pixelközép-konvencióval)
        pts = (pts + 0.5) / s - 0.5
        feats[photo.name] = (pts, desc, colors)
        log(f"  [{i+1}/{len(project.photos)}] {photo.name}: {len(kps)} pont")
    return feats


def select_pairs(project, max_neighbors=8, log=print):
    """Illesztendő képpárok kiválasztása.

    Ha van GPS: minden képhez a `max_neighbors` legközelebbi szomszéd.
    Ha nincs: minden pár (kis képszámnál), különben csúszóablak.
    """
    n = len(project.photos)
    names = [p.name for p in project.photos]
    has_gps = all(p.gps_eov is not None for p in project.photos)
    pairs = set()
    if has_gps and n > 3:
        pos = np.array([p.gps_eov[:2] for p in project.photos])
        for i in range(n):
            d = np.linalg.norm(pos - pos[i], axis=1)
            order = np.argsort(d)
            for j in order[1:max_neighbors + 1]:
                pairs.add((min(i, int(j)), max(i, int(j))))
    elif n <= 30:
        for i in range(n):
            for j in range(i + 1, n):
                pairs.add((i, j))
    else:
        for i in range(n):
            for j in range(i + 1, min(n, i + 1 + max_neighbors)):
                pairs.add((i, j))
    pairs = sorted(pairs)
    log(f"  {len(pairs)} képpár kijelölve ({'GPS-szomszédság' if has_gps and n > 3 else 'teljes'} alapján)")
    return [(names[i], names[j]) for i, j in pairs]


def match_pairs(project, feats, pairs, ratio=0.8, min_inliers=25, log=print):
    """Párok illesztése: FLANN + arányteszt + geometriai (F-mátrix RANSAC) szűrés.

    Visszatér: dict (nameA, nameB) -> Mx2 int index-pár tömb (A-beli, B-beli).
    """
    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher({"algorithm": FLANN_INDEX_KDTREE, "trees": 5},
                                  {"checks": 64})
    out = {}
    for k, (na, nb) in enumerate(pairs):
        pa, da, _ = feats[na]
        pb, db, _ = feats[nb]
        if len(da) < 8 or len(db) < 8:
            continue
        raw = flann.knnMatch(da, db, k=2)
        good = []
        for m_n in raw:
            if len(m_n) == 2 and m_n[0].distance < ratio * m_n[1].distance:
                good.append((m_n[0].queryIdx, m_n[0].trainIdx))
        if len(good) < min_inliers:
            continue
        good = np.array(good, np.int64)
        ptsA = pa[good[:, 0]].astype(np.float64)
        ptsB = pb[good[:, 1]].astype(np.float64)
        F, mask = cv2.findFundamentalMat(ptsA, ptsB, cv2.FM_RANSAC, 3.0, 0.999)
        if F is None or mask is None:
            continue
        inl = mask.ravel().astype(bool)
        if inl.sum() < min_inliers:
            continue
        out[(na, nb)] = good[inl]
        log(f"  [{k+1}/{len(pairs)}] {na} - {nb}: {inl.sum()} illesztés")
    return out


def build_tracks(feats, matches, min_track_len=2, log=print):
    """Kötőpont-láncok (track) építése union-find-dal.

    Visszatér: list[dict photo_name -> feat_idx], képenként max 1 megfigyeléssel.
    """
    parent = {}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (na, nb), idx in matches.items():
        for ia, ib in idx:
            a = (na, int(ia)); b = (nb, int(ib))
            if a not in parent:
                parent[a] = a
            if b not in parent:
                parent[b] = b
            union(a, b)

    groups = {}
    for key in parent:
        groups.setdefault(find(key), []).append(key)

    tracks = []
    dropped = 0
    for members in groups.values():
        track = {}
        ok = True
        for name, idx in members:
            if name in track:      # ugyanabban a képben két pont -> inkonzisztens
                ok = False
                break
            track[name] = idx
        if ok and len(track) >= min_track_len:
            tracks.append(track)
        else:
            dropped += 1
    log(f"  {len(tracks)} kötőpont-lánc ({dropped} inkonzisztens eldobva)")
    return tracks
