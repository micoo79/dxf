"""PyShape — nyílt fotogrammetriai munkafolyamat (Metashape-szerű) Pythonban.

Munkafolyamat:
  1. Képek betöltése (EXIF GPS -> EOV / EPSG:23700)
  2. Align photos (SIFT + inkrementális SfM + bundle adjustment)
  3. Georeferálás kamera-GPS alapján, majd GCP-kkel pontosítva
  4. GCP-k betöltése fájlból + kézi azonosítás a képeken
  5. Dense cloud (sztereó mélységtérképek fúziója)
  6. Mesh / DSM
  7. Orthofotó + GeoTIFF export
"""

__version__ = "1.0.0"
APP_NAME = "PyShape"
