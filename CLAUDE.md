# PyShape — projektkontextus (Claude-nak)

Ez a fájl a projekt teljes átadó dokumentációja. Ha most nyitod meg ezt a
projektet: olvasd végig, ez pótolja a korábbi beszélgetés kontextusát.

## Mi ez?

**PyShape**: Agisoft Metashape-szerű fotogrammetriai program tisztán
Pythonban, drónos légi felmérésekhez, **EOV (EPSG:23700)** vetületben.
A teljes lánc működik és szintetikus adatokon end-to-end tesztelt:

képek betöltése (EXIF GPS→EOV) → align photos (SIFT + inkrementális SfM +
bundle adjustment) → georeferálás (GPS, majd GCP) → GCP-k fájlból + kézi
képi bemérés a GUI-ban → optimize (GCP-kényszeres BA) → dense cloud (SGBM
több-partneres fúzióval) → DSM/mesh → orthofotó → **GeoTIFF export**.

A felhasználó (Miklós, geolink3d.hu) földmérő; magyarul kommunikálj vele,
egyszerűen, technikai zsargon nélkül. Nem programozó és nem ismeri a gitet —
a "branch/commit" fogalmakat kerüld, helyette: "elmentettem/feltöltöttem".

## Hol tart a projekt? (2026-08-20)

- A teljes pipeline kész és a szintetikus ön-teszten 10/10 PASS:
  `python tools/selftest.py` (kb. 4-6 perc) — GCP RMS ~3 cm, dense magassági
  medián ±2 cm, a GCP-jelek pixelre a helyükön az ortho GeoTIFF-en.
- GUI (Tkinter) működik: munkafolyamat-menü, folyamatjelző + állapotsor,
  GCP-panel (GCP kiválasztás → képlista, ahol látszik → kattintásos mérés).
- **A valódi drónfotós teszt még hátravan** — ez a következő lépés a
  felhasználó gépén. Számíts rá, hogy valós adatokon hangolni kell
  (küszöbök, memóriahasználat nagy képeknél, RAW/DNG nem támogatott).

## A felhasználó gépén eddig előjött gondok (Windows)

1. **Free-threaded Python 3.13t** volt telepítve → numpy/opencv nem megy
   vele. Az `INDITAS_Windows.bat` már automatikusan normál (GIL-es) 3.10+
   Pythont keres és kihagyja a "t" változatot.
2. A hibás első futás **sérült csomagokat** hagyott a site-packages-ben
   (pl. `pyproj._context` hiányzott) → a bat importtal ellenőrzi a csomagok
   épségét és szükség esetén `--force-reinstall`-lal javít.
3. **Ékezetes útvonalak** (C:\...\Képek\...): cv2.imread nem olvassa →
   `pyshape/imio.py: imread()` bájt-dekódolós fallbackkel; MINDIG ezt
   használd cv2.imread helyett.
4. Kérés volt: **mindig legyen látható állapot-visszajelzés** — minden új
   hosszú művelethez adj progress-callbacket és naplósort (lásd
   `gui/app.py: _run_task/_set_busy`, `_progress_cb`).

## Futtatás / tesztelés

- GUI: `python -m pyshape gui` (vagy dupla katt: `INDITAS_Windows.bat`)
- CLI teljes lánc: lásd `README.md` "Parancssori használat"
- Önteszt (szintetikus, ground-truth ellenőrzéssel): `python tools/selftest.py`
- Gyors szintetikus adat: `python tools/make_synthetic_dataset.py <mappa> --images 12`
- Függőségek: `requirements.txt` (numpy, scipy, opencv-python, pyproj,
  rasterio, Pillow) + tkinter. GPU nem kell.

## Architektúra (pyshape/)

| Modul | Feladat |
|---|---|
| `project.py` | adatmodell: Photo (R: world→cam, C: középpont), CameraModel (f,cx,cy,k1,k2), Marker (GCP), Project mentés/betöltés (`.pshape` mappa: project.json + sparse.npz) |
| `geodesy.py` | WGS84↔EOV (pyproj); world = EOV − origin (kis számok a BA-nak) |
| `exif.py`, `loader.py` | EXIF GPS/fókusz, képek betöltése, referencia-CSV |
| `features.py` | SIFT, GPS-szomszédos párkijelölés, FLANN+F-RANSAC illesztés, track-építés |
| `sfm.py` | inkrementális SfM: E/H-inicializálás, PnP, DLT-trianguláció; 15 kép fölött lokális BA (utolsó 8 kamera) + periodikus globális |
| `ba.py` | scipy sparse least_squares BA; fix pontok (GCP), kamera-priorok (GPS), megfigyelés-súlyok |
| `georef.py` | Umeyama-hasonlóság + sík-kétértelműség feloldás (terep a kamerák ALATT — légi feltevés) |
| `gcp.py` | GCP-fájl (nev;E;N;H), láthatóság-előrejelzés, GCP-kényszeres optimize |
| `dense.py` | sztereórektifikáció + SGBM; képenként ≤3 partner, mélységfúzió mediánnal + egyezésszűrés. FONTOS: K-skálázás pixelközép-korrekcióval (cx_s = s·cx + (s−1)/2) — enélkül páronkénti szisztematikus hiba! |
| `meshing.py` | DSM (cellánkénti medián, tüskeszűrés, lyukkitöltés), 2.5D TIN mesh, DSM GeoTIFF |
| `ortho.py` | DSM-alapú ortorektifikáció, képközép-súlyozott keverés, RGBA GeoTIFF EPSG:23700 |
| `imio.py` | unicode-biztos imread — cv2.imread TILOS közvetlenül |
| `cli.py` | alparancsok: new/align/georef/gcp-load/gcp-measure/optimize/dense/mesh/ortho/status |
| `gui/` | Tkinter: app.py (főablak, worker-szál + queue-s napló), imageview.py (zoom/pan/kattintás) |

## Fontos tervezési döntések / buktatók

- Kamerapóz: `x_cam = R @ (X − C)`; BA-ban rvec+tvec, t = −R·C.
- Minden kulcspont/mérés TELJES felbontású pixelben tárolódik.
- Kicsinyítésnél pixelközép-konvenció: `u_s = s·u_f + (s−1)/2` — ezt a
  features.py és dense.py már jól csinálja, új kódban is így kell.
- Nadír blokkoknál f↔magasság korreláció miatt a kameramagasság ~1 m-t
  "csúszhat" — a TALAJ (GCP) pontosság a mérvadó, ez normális jelenség.
- A georef sík-degenerációját a terep-a-kamerák-alatt ellenőrzés oldja fel
  (`similarity_terrain_aware`, cam_pts paraméterrel).
- GUI-ban hosszú műveletet csak `_run_task`-on át indíts (busy-jelzés,
  menü-tiltás, hibadialógus); a worker-szálból UI-t közvetlenül ne piszkálj,
  csak `self.log`/`self._progress_cb`.

## Valószínű következő feladatok

1. Valódi drónfotókkal próba a felhasználó gépén (több ezer × 20MP kép:
   memória- és sebességhangolás, esetleg feature-cache lemezre).
2. RAW/DNG támogatás ha kell (rawpy), maszkolás, vízfelületek kezelése.
3. Ortho takarásvizsgálat (DSM-raycast) ferde képeknél.
4. Jelentés-export (GCP-hibatáblázat CSV/PDF).

## Munkamódszer

- A kód magyarul kommentelt/naplózó; a felhasználói üzenetek magyarul.
- Változtatás után futtasd le legalább: `python -m py_compile pyshape/*.py
  pyshape/gui/*.py`, és ha a mag változott, a `tools/selftest.py`-t.
- A GitHub-tárhely: micoo79/dxf, a munka a
  `claude/metashape-python-clone-a9gb2j` ágon van. A repó `dxf-viewer.html`
  fájlja egy ettől FÜGGETLEN régi eszköz — ne nyúlj hozzá.
