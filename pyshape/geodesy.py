"""Geodéziai transzformációk: WGS84 <-> EOV (EPSG:23700).

Az EOV vetületi koordinátarendszer méterben mér, ezért a belső ("world")
koordinátarendszerünk egy lokális EOV-eltolt rendszer: world = EOV - origin.
Így a bundle adjustment kis számokkal dolgozik, az export pedig visszatolja
az origó hozzáadásával.

Megjegyzés a magasságról: az EXIF GPS-magasság ellipszoidi (WGS84), az EOV-hoz
tartozó balti magasság ettől Magyarországon ~40-45 m-rel tér el (geoidunduláció).
GCP-k használatakor ez kiesik (a GCP-k adják a valódi magassági keretet);
GCP nélkül a `geoid_offset` beállítással korrigálható.
"""

from pyproj import Transformer

_wgs_to_eov = None
_eov_to_wgs = None


def wgs84_to_eov(lon, lat, alt=0.0, geoid_offset=0.0):
    """WGS84 (fok) -> EOV (m). Visszatér: (E, N, H)."""
    global _wgs_to_eov
    if _wgs_to_eov is None:
        _wgs_to_eov = Transformer.from_crs("EPSG:4326", "EPSG:23700", always_xy=True)
    e, n = _wgs_to_eov.transform(lon, lat)
    return e, n, alt - geoid_offset


def eov_to_wgs84(e, n, h=0.0, geoid_offset=0.0):
    """EOV (m) -> WGS84 (fok). Visszatér: (lon, lat, alt)."""
    global _eov_to_wgs
    if _eov_to_wgs is None:
        _eov_to_wgs = Transformer.from_crs("EPSG:23700", "EPSG:4326", always_xy=True)
    lon, lat = _eov_to_wgs.transform(e, n)
    return lon, lat, h + geoid_offset


EOV_CRS = "EPSG:23700"
