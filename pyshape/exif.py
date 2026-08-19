"""EXIF-olvasás: GPS pozíció és fókusztávolság kinyerése a képekből."""

import os

from PIL import Image, ExifTags

GPS_TAG = None
for _t, _n in ExifTags.TAGS.items():
    if _n == "GPSInfo":
        GPS_TAG = _t
        break

_GPS_NAMES = {v: k for k, v in ExifTags.GPSTAGS.items()}


def _rat(v):
    try:
        return float(v)
    except TypeError:
        # régi Pillow: (num, den) tuple
        return v[0] / v[1] if v[1] else 0.0


def _dms_to_deg(dms, ref):
    d = _rat(dms[0]) + _rat(dms[1]) / 60.0 + _rat(dms[2]) / 3600.0
    if ref in ("S", "W", b"S", b"W"):
        d = -d
    return d


def read_exif(path):
    """Kép EXIF-adatainak kiolvasása.

    Visszatér dict:
      width, height        - képméret pixelben
      lat, lon, alt        - WGS84 (fok, fok, m) vagy None
      f35                  - 35 mm-egyenértékű fókusz (mm) vagy None
      focal_mm             - valódi fókusz (mm) vagy None
      make, model
    """
    out = {"width": None, "height": None, "lat": None, "lon": None,
           "alt": None, "f35": None, "focal_mm": None, "make": "", "model": ""}
    with Image.open(path) as im:
        out["width"], out["height"] = im.size
        try:
            exif = im.getexif()
        except Exception:
            return out
        if not exif:
            return out

        def get_named(name):
            for tag_id, tag_name in ExifTags.TAGS.items():
                if tag_name == name and tag_id in exif:
                    return exif[tag_id]
            return None

        out["make"] = str(get_named("Make") or "").strip("\x00 ")
        out["model"] = str(get_named("Model") or "").strip("\x00 ")

        # Exif al-IFD (fókusz)
        try:
            sub = exif.get_ifd(0x8769)
        except Exception:
            sub = {}
        for tag_id, val in (sub or {}).items():
            name = ExifTags.TAGS.get(tag_id, "")
            if name == "FocalLengthIn35mmFilm":
                try:
                    out["f35"] = float(val)
                except Exception:
                    pass
            elif name == "FocalLength":
                try:
                    out["focal_mm"] = _rat(val)
                except Exception:
                    pass

        # GPS IFD
        try:
            gps = exif.get_ifd(0x8825)
        except Exception:
            gps = {}
        if gps:
            named = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}
            try:
                if "GPSLatitude" in named and "GPSLongitude" in named:
                    out["lat"] = _dms_to_deg(named["GPSLatitude"],
                                             named.get("GPSLatitudeRef", "N"))
                    out["lon"] = _dms_to_deg(named["GPSLongitude"],
                                             named.get("GPSLongitudeRef", "E"))
                if "GPSAltitude" in named:
                    alt = _rat(named["GPSAltitude"])
                    ref = named.get("GPSAltitudeRef", 0)
                    if isinstance(ref, bytes):
                        ref = ref[0] if ref else 0
                    if ref == 1:
                        alt = -alt
                    out["alt"] = alt
            except Exception:
                pass
    return out


def estimate_focal_px(exif_info, width):
    """Fókusztávolság becslése pixelben az EXIF-ből; None ha nincs adat."""
    if exif_info.get("f35"):
        return exif_info["f35"] / 36.0 * width
    return None
