"""PyShape grafikus felület (Tkinter).

Elrendezés:
  - bal oldalt: Fényképek / GCP-k fülek
  - középen: képnéző (zoom: görgő, mozgatás: húzás, mérés: kattintás)
  - alul: napló + folyamatjelző
  - Munkafolyamat menü: a teljes feldolgozási lánc

GCP kézi azonosítás: a GCP-k fülön egy pontot kiválasztva megjelenik azoknak
a képeknek a listája, amelyeken a pont (előrejelzés szerint) látszik; egy
képet megnyitva a becsült hely sárga kereszttel jelenik meg, kattintással
rögzíthető / módosítható a mérés (jobb kattintás: törlés).
"""

import os
import queue
import threading
import traceback

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np
from PIL import Image

from .. import APP_NAME, __version__
from ..project import Project
from .imageview import ImageView

PRED_COLOR = "#ffd633"     # becsült hely: sárga
MEAS_COLOR = "#33ff55"     # mérés: zöld
OTHER_COLOR = "#66aaff"    # más GCP mérése ezen a képen: kék


class App(tk.Tk):
    def __init__(self, project_path=None):
        super().__init__()
        self.title(f"{APP_NAME} {__version__}")
        self.geometry("1480x900")
        self.prj = None
        self.current_photo = None
        self.current_marker = None
        self.marking_mode = False
        self._queue = queue.Queue()
        self._worker = None

        self._build_menu()
        self._build_layout()
        self.after(100, self._poll_queue)

        if project_path:
            self._open_project(project_path)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ================================================================== UI

    def _build_menu(self):
        m = tk.Menu(self)
        fm = tk.Menu(m, tearoff=0)
        fm.add_command(label="Új projekt...", command=self.mi_new_project)
        fm.add_command(label="Projekt megnyitása...", command=self.mi_open_project)
        fm.add_command(label="Mentés", command=self.mi_save, accelerator="Ctrl+S")
        fm.add_separator()
        fm.add_command(label="Kilépés", command=self._on_close)
        m.add_cascade(label="Fájl", menu=fm)

        wm = tk.Menu(m, tearoff=0)
        wm.add_command(label="1. Képek hozzáadása...", command=self.mi_add_photos)
        wm.add_command(label="    Kamerapozíció-referencia betöltése (CSV)...",
                       command=self.mi_load_reference)
        wm.add_command(label="2. Align photos (képek beállítása)",
                       command=self.mi_align)
        wm.add_command(label="3. Georeferálás kamera-GPS alapján",
                       command=self.mi_georef)
        wm.add_command(label="4. GCP-k betöltése fájlból...", command=self.mi_load_gcps)
        wm.add_command(label="5. Kamerák optimalizálása (GCP-kkel)",
                       command=self.mi_optimize)
        wm.add_command(label="6. Dense cloud építése", command=self.mi_dense)
        wm.add_command(label="7. Mesh / DSM építése", command=self.mi_mesh)
        wm.add_command(label="8. Orthofotó + GeoTIFF export", command=self.mi_ortho)
        m.add_cascade(label="Munkafolyamat", menu=wm)
        self.workflow_menu = wm

        vm = tk.Menu(m, tearoff=0)
        vm.add_command(label="Orthofotó megnyitása a nézőben",
                       command=self.mi_view_ortho)
        vm.add_command(label="DSM megnyitása a nézőben", command=self.mi_view_dsm)
        m.add_cascade(label="Nézet", menu=vm)

        hm = tk.Menu(m, tearoff=0)
        hm.add_command(label="Névjegy", command=lambda: messagebox.showinfo(
            APP_NAME,
            f"{APP_NAME} {__version__}\n\nNyílt fotogrammetriai munkafolyamat\n"
            "EOV (EPSG:23700) támogatással.\n\n"
            "Munkafolyamat: képek -> align -> GCP -> dense -> mesh -> ortho"))
        m.add_cascade(label="Súgó", menu=hm)
        self.config(menu=m)
        self.bind("<Control-s>", lambda e: self.mi_save())

    def _build_layout(self):
        outer = ttk.Panedwindow(self, orient="horizontal")
        outer.pack(fill="both", expand=True)

        # ---- bal panel: fülek
        left = ttk.Frame(outer, width=380)
        outer.add(left, weight=0)
        nb = ttk.Notebook(left)
        nb.pack(fill="both", expand=True)

        # Fényképek fül
        ptab = ttk.Frame(nb)
        nb.add(ptab, text="Fényképek")
        self.photo_tree = ttk.Treeview(
            ptab, columns=("eov", "ok"), show="tree headings", selectmode="browse")
        self.photo_tree.heading("#0", text="Kép")
        self.photo_tree.heading("eov", text="EOV (E, N, H)")
        self.photo_tree.heading("ok", text="Beállt")
        self.photo_tree.column("#0", width=130)
        self.photo_tree.column("eov", width=180)
        self.photo_tree.column("ok", width=50, anchor="center")
        self.photo_tree.pack(fill="both", expand=True)
        self.photo_tree.bind("<Double-1>", self._on_photo_open)

        # GCP fül
        gtab = ttk.Frame(nb)
        nb.add(gtab, text="GCP-k")
        gbtns = ttk.Frame(gtab)
        gbtns.pack(fill="x")
        ttk.Button(gbtns, text="GCP-fájl betöltése...",
                   command=self.mi_load_gcps).pack(side="left", padx=2, pady=2)
        ttk.Button(gbtns, text="Optimalizálás",
                   command=self.mi_optimize).pack(side="left", padx=2, pady=2)
        self.gcp_tree = ttk.Treeview(
            gtab, columns=("eov", "n"), show="tree headings",
            selectmode="browse", height=8)
        self.gcp_tree.heading("#0", text="GCP")
        self.gcp_tree.heading("eov", text="EOV (E, N, H)")
        self.gcp_tree.heading("n", text="Mérés")
        self.gcp_tree.column("#0", width=90)
        self.gcp_tree.column("eov", width=190)
        self.gcp_tree.column("n", width=50, anchor="center")
        self.gcp_tree.pack(fill="both", expand=False)
        self.gcp_tree.bind("<<TreeviewSelect>>", self._on_gcp_select)

        ttk.Label(gtab, text="Képek, amelyeken a kiválasztott GCP látszik:"
                  ).pack(fill="x", padx=2, pady=(6, 0))
        self.gcp_img_tree = ttk.Treeview(
            gtab, columns=("state", "pos"), show="tree headings",
            selectmode="browse")
        self.gcp_img_tree.heading("#0", text="Kép")
        self.gcp_img_tree.heading("state", text="Állapot")
        self.gcp_img_tree.heading("pos", text="Becsült (u,v)")
        self.gcp_img_tree.column("#0", width=120)
        self.gcp_img_tree.column("state", width=90, anchor="center")
        self.gcp_img_tree.column("pos", width=110, anchor="center")
        self.gcp_img_tree.pack(fill="both", expand=True)
        self.gcp_img_tree.bind("<Double-1>", self._on_gcp_image_open)
        ttk.Label(gtab, text=("Kattintás a képen: mérés rögzítése\n"
                              "Jobb kattintás: mérés törlése"),
                  foreground="#666").pack(fill="x", padx=2, pady=2)

        # ---- jobb oldal: néző + napló
        right = ttk.Frame(outer)
        outer.add(right, weight=1)
        topbar = ttk.Frame(right)
        topbar.pack(fill="x")
        self.viewer_label = ttk.Label(topbar, text="(nincs kép)")
        self.viewer_label.pack(side="left", padx=4)
        ttk.Button(topbar, text="Teljes kép", command=lambda: self.viewer.fit()
                   ).pack(side="right", padx=2, pady=2)

        vpane = ttk.Panedwindow(right, orient="vertical")
        vpane.pack(fill="both", expand=True)
        self.viewer = ImageView(vpane)
        vpane.add(self.viewer, weight=3)
        self.viewer.on_click = self._on_viewer_click
        self.viewer.on_right_click = self._on_viewer_right_click

        bottom = ttk.Frame(vpane)
        vpane.add(bottom, weight=1)
        self.progress = ttk.Progressbar(bottom, mode="determinate", maximum=1.0)
        self.progress.pack(fill="x")
        self.log_text = tk.Text(bottom, height=8, state="disabled",
                                bg="#111418", fg="#c8d0d8", wrap="word")
        sb = ttk.Scrollbar(bottom, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)

        self.status = ttk.Label(self, text="Nincs projekt betöltve.", anchor="w")
        self.status.pack(fill="x")

    # ============================================================== segédek

    def log(self, msg):
        self._queue.put(("log", str(msg)))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "log":
                    self.log_text.configure(state="normal")
                    self.log_text.insert("end", payload + "\n")
                    self.log_text.see("end")
                    self.log_text.configure(state="disabled")
                elif kind == "progress":
                    self.progress["value"] = payload
                elif kind == "done":
                    self.progress["value"] = 0
                    self._task_finished(payload)
                elif kind == "error":
                    self.progress["value"] = 0
                    self._task_finished(None)
                    messagebox.showerror("Hiba", payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _run_task(self, name, fn, on_done=None):
        if self._worker is not None and self._worker.is_alive():
            messagebox.showwarning(APP_NAME, "Már fut egy művelet.")
            return
        self._task_done_cb = on_done
        self.log(f"=== {name} ===")

        def wrapper():
            try:
                fn()
                self._queue.put(("done", name))
            except Exception as e:
                traceback.print_exc()
                self._queue.put(("error", f"{name}: {e}"))

        self._worker = threading.Thread(target=wrapper, daemon=True)
        self._worker.start()

    def _task_finished(self, name):
        self._refresh_all()
        if name and self._task_done_cb:
            cb, self._task_done_cb = self._task_done_cb, None
            cb()
        if name:
            self.log(f"=== {name} kész ===")

    def _need_project(self):
        if self.prj is None:
            messagebox.showwarning(APP_NAME, "Előbb hozz létre / nyiss meg egy projektet.")
            return True
        return False

    def _progress_cb(self, frac):
        self._queue.put(("progress", frac))

    # ======================================================== frissítések

    def _refresh_all(self):
        self._refresh_photos()
        self._refresh_gcps()
        self._refresh_status()

    def _refresh_photos(self):
        t = self.photo_tree
        t.delete(*t.get_children())
        if not self.prj:
            return
        for p in self.prj.photos:
            eov = ""
            if p.gps_eov:
                eov = f"{p.gps_eov[0]:.1f}, {p.gps_eov[1]:.1f}, {p.gps_eov[2]:.0f}"
            t.insert("", "end", iid=p.name, text=p.name,
                     values=(eov, "✓" if p.aligned else ""))

    def _refresh_gcps(self):
        t = self.gcp_tree
        sel = t.selection()
        t.delete(*t.get_children())
        if not self.prj:
            return
        for mk in self.prj.markers:
            t.insert("", "end", iid=mk.name, text=mk.name,
                     values=(f"{mk.eov[0]:.2f}, {mk.eov[1]:.2f}, {mk.eov[2]:.2f}",
                             len(mk.pixels)))
        if sel and t.exists(sel[0]):
            t.selection_set(sel[0])
        self._refresh_gcp_images()

    def _refresh_gcp_images(self):
        from ..gcp import predict_marker_images
        t = self.gcp_img_tree
        t.delete(*t.get_children())
        self.current_marker = None
        if not self.prj:
            return
        sel = self.gcp_tree.selection()
        if not sel:
            return
        mk = next((m for m in self.prj.markers if m.name == sel[0]), None)
        if mk is None:
            return
        self.current_marker = mk
        preds = predict_marker_images(self.prj, mk)
        listed = set()
        for photo, u, v, _ in preds:
            state = "bemérve" if photo.name in mk.pixels else "-"
            t.insert("", "end", iid=photo.name, text=photo.name,
                     values=(state, f"{u:.0f}, {v:.0f}"))
            listed.add(photo.name)
        # mérések olyan képeken, amelyek nem szerepelnek az előrejelzésben
        for name in mk.pixels:
            if name not in listed and self.prj.photo_by_name(name):
                t.insert("", "end", iid=name, text=name,
                         values=("bemérve", ""))
                listed.add(name)
        if not preds:
            # georeferálás nélkül nincs előrejelzés — az összes kép felajánlása,
            # hogy a mérés kézzel akkor is elvégezhető legyen
            if not self.prj.georeferenced:
                self.log("Nincs georeferálás — a lista az összes képet mutatja "
                         "(a becsült hely nem számítható).")
            for p in self.prj.photos:
                if p.name not in listed:
                    t.insert("", "end", iid=p.name, text=p.name,
                             values=("-", ""))

    def _refresh_status(self):
        if not self.prj:
            self.status.config(text="Nincs projekt betöltve.")
            return
        n_al = len(self.prj.aligned_photos())
        n_pt = 0 if self.prj.points is None else len(self.prj.points)
        n_meas = sum(1 for m in self.prj.markers if len(m.pixels) >= 2)
        self.status.config(text=(
            f"Projekt: {self.prj.path}   |   képek: {len(self.prj.photos)} "
            f"(beállt: {n_al})   |   kötőpontok: {n_pt:,}   |   "
            f"GCP: {len(self.prj.markers)} (bemérve: {n_meas})   |   "
            f"georeferált: {'igen' if self.prj.georeferenced else 'nem'}"))

    # ========================================================== képmegnyitás

    def _open_photo_in_viewer(self, photo, marking=False, center_uv=None):
        try:
            img = Image.open(photo.path)
            img.load()
        except Exception as e:
            messagebox.showerror(APP_NAME, f"Nem nyitható meg: {photo.path}\n{e}")
            return
        self.current_photo = photo
        self.marking_mode = marking
        self.viewer_label.config(text=photo.name + ("   [GCP-mérés mód]" if marking else ""))
        self.viewer.set_image(img)
        self._update_viewer_markers()
        if center_uv is not None:
            self.viewer.center_on(center_uv[0], center_uv[1], zoom=2.0)

    def _update_viewer_markers(self):
        markers = []
        if self.prj and self.current_photo:
            pname = self.current_photo.name
            for mk in self.prj.markers:
                if pname in mk.pixels:
                    u, v = mk.pixels[pname]
                    color = MEAS_COLOR if mk is self.current_marker else OTHER_COLOR
                    markers.append((u, v, color, mk.name))
            # a kiválasztott GCP becsült helye
            if (self.current_marker and self.prj.georeferenced
                    and self.current_photo.aligned
                    and pname not in self.current_marker.pixels):
                from ..gcp import predict_marker_images
                for photo, u, v, _ in predict_marker_images(self.prj,
                                                            self.current_marker):
                    if photo.name == pname:
                        markers.append((u, v, PRED_COLOR,
                                        self.current_marker.name + " (becsült)"))
                        break
        self.viewer.set_markers(markers)

    def _on_photo_open(self, _e):
        sel = self.photo_tree.selection()
        if sel and self.prj:
            photo = self.prj.photo_by_name(sel[0])
            if photo:
                self._open_photo_in_viewer(photo, marking=False)

    def _on_gcp_select(self, _e):
        self._refresh_gcp_images()

    def _on_gcp_image_open(self, _e):
        sel = self.gcp_img_tree.selection()
        if not (sel and self.prj and self.current_marker):
            return
        photo = self.prj.photo_by_name(sel[0])
        if not photo:
            return
        mk = self.current_marker
        if photo.name in mk.pixels:
            uv = mk.pixels[photo.name]
        else:
            vals = self.gcp_img_tree.item(sel[0], "values")
            try:
                uv = tuple(float(x) for x in vals[1].split(","))
            except Exception:
                uv = None
        self._open_photo_in_viewer(photo, marking=True, center_uv=uv)

    def _on_viewer_click(self, u, v):
        if not (self.marking_mode and self.prj and self.current_marker
                and self.current_photo):
            return
        mk = self.current_marker
        mk.pixels[self.current_photo.name] = (float(u), float(v))
        self.log(f"{mk.name} bemérve: {self.current_photo.name} "
                 f"({u:.1f}, {v:.1f})")
        self.prj.save()
        self._refresh_gcps()
        self._update_viewer_markers()

    def _on_viewer_right_click(self, u, v):
        if not (self.marking_mode and self.prj and self.current_marker
                and self.current_photo):
            return
        mk = self.current_marker
        if self.current_photo.name in mk.pixels:
            del mk.pixels[self.current_photo.name]
            self.log(f"{mk.name} mérése törölve: {self.current_photo.name}")
            self.prj.save()
            self._refresh_gcps()
            self._update_viewer_markers()

    # ========================================================== menüakciók

    def mi_new_project(self):
        path = filedialog.asksaveasfilename(
            title="Új projekt helye", defaultextension=".pshape",
            filetypes=[("PyShape projekt", "*.pshape")])
        if not path:
            return
        self.prj = Project(path)
        os.makedirs(path, exist_ok=True)
        self.prj.save()
        self._refresh_all()
        self.log(f"Új projekt: {path}")

    def mi_open_project(self):
        path = filedialog.askdirectory(title="Projektmappa (.pshape) megnyitása")
        if path:
            self._open_project(path)

    def _open_project(self, path):
        try:
            self.prj = Project.load(path)
        except Exception as e:
            messagebox.showerror(APP_NAME, f"Nem nyitható meg: {e}")
            return
        self._refresh_all()
        self.log(f"Projekt megnyitva: {path} ({len(self.prj.photos)} kép)")

    def mi_save(self):
        if self._need_project():
            return
        self.prj.save()
        self.log("Projekt elmentve.")

    def mi_add_photos(self):
        if self._need_project():
            return
        paths = filedialog.askopenfilenames(
            title="Képek kiválasztása",
            filetypes=[("Képek", "*.jpg *.jpeg *.png *.tif *.tiff"),
                       ("Minden fájl", "*.*")])
        if not paths:
            return
        from ..loader import add_photos

        def task():
            add_photos(self.prj, list(paths), log=self.log)
            self.prj.save()
        self._run_task("Képek hozzáadása", task)

    def mi_load_reference(self):
        if self._need_project():
            return
        path = filedialog.askopenfilename(
            title="Referencia CSV (kep;E;N;H)",
            filetypes=[("CSV", "*.csv *.txt"), ("Minden fájl", "*.*")])
        if not path:
            return
        from ..loader import load_reference_csv
        ref = load_reference_csv(path)
        n = 0
        for p in self.prj.photos:
            if p.name in ref:
                p.gps_eov = ref[p.name]
                n += 1
        if self.prj.origin is None and n:
            pts = np.array([p.gps_eov for p in self.prj.photos if p.gps_eov])
            self.prj.origin = (float(round(pts[:, 0].mean())),
                               float(round(pts[:, 1].mean())), 0.0)
        self.prj.save()
        self.log(f"Referencia betöltve {n} képhez.")
        self._refresh_all()

    def mi_align(self):
        if self._need_project():
            return
        q = _ask_choice(self, "Align photos", "Minőség:",
                        ["highest", "high", "medium", "low"], "high")
        if not q:
            return
        from ..sfm import align_photos
        from ..georef import georeference_from_gps

        def task():
            align_photos(self.prj, quality=q, log=self.log)
            if sum(1 for p in self.prj.photos
                   if p.aligned and p.gps_eov) >= 3:
                georeference_from_gps(self.prj, log=self.log)
            self.prj.save()
        self._run_task("Align photos", task)

    def mi_georef(self):
        if self._need_project():
            return
        from ..georef import georeference_from_gps

        def task():
            georeference_from_gps(self.prj, log=self.log)
            self.prj.save()
        self._run_task("Georeferálás (GPS)", task)

    def mi_load_gcps(self):
        if self._need_project():
            return
        path = filedialog.askopenfilename(
            title="GCP-fájl (nev;E;N;H)",
            filetypes=[("CSV/TXT", "*.csv *.txt"), ("Minden fájl", "*.*")])
        if not path:
            return
        from ..gcp import load_gcp_file
        load_gcp_file(self.prj, path, log=self.log)
        self.prj.save()
        self._refresh_all()

    def mi_optimize(self):
        if self._need_project():
            return
        from ..gcp import optimize_with_gcps

        def task():
            optimize_with_gcps(self.prj, log=self.log)
            self.prj.save()
        self._run_task("Kamerák optimalizálása (GCP)", task)

    def mi_dense(self):
        if self._need_project():
            return
        q = _ask_choice(self, "Dense cloud", "Minőség:",
                        ["high", "medium", "low"], "medium")
        if not q:
            return
        from ..dense import build_dense_cloud

        def task():
            out = os.path.join(self.prj.path, "dense.ply")
            build_dense_cloud(self.prj, quality=q, out_path=out, log=self.log,
                              progress=self._progress_cb)
            self.prj.save()
        self._run_task("Dense cloud", task)

    def mi_mesh(self):
        if self._need_project():
            return
        from ..io_ply import read_ply_points
        from ..meshing import (build_dsm, build_mesh, save_dsm_geotiff,
                               save_mesh_ply)

        def task():
            dense_path = os.path.join(self.prj.path, "dense.ply")
            if not os.path.exists(dense_path):
                raise RuntimeError("Előbb futtasd a Dense cloud lépést.")
            pts, col = read_ply_points(dense_path)
            dsm = build_dsm(pts, col, log=self.log)
            save_dsm_geotiff(os.path.join(self.prj.path, "dsm.tif"), dsm,
                             self.prj.origin, log=self.log)
            verts, faces, vcol = build_mesh(dsm, log=self.log)
            save_mesh_ply(os.path.join(self.prj.path, "mesh.ply"),
                          verts, faces, vcol, origin=self.prj.origin)
            self.log("Mesh (PLY, EOV) és DSM (GeoTIFF) elmentve a projektmappába.")
        self._run_task("Mesh / DSM", task)

    def mi_ortho(self):
        if self._need_project():
            return
        gsd_s = simpledialog.askstring(
            "Orthofotó", "Felbontás méterben (üres = automatikus):",
            parent=self)
        if gsd_s is None:
            return
        try:
            gsd = float(gsd_s.replace(",", ".")) if gsd_s.strip() else None
        except ValueError:
            messagebox.showerror(APP_NAME, "Érvénytelen felbontás.")
            return
        out = filedialog.asksaveasfilename(
            title="GeoTIFF mentése", defaultextension=".tif",
            initialfile="ortho.tif",
            filetypes=[("GeoTIFF", "*.tif")])
        if not out:
            return
        from ..cli import load_dsm_from_project
        from ..ortho import build_orthophoto

        def task():
            dsm = load_dsm_from_project(self.prj)
            if dsm is None:
                raise RuntimeError("Előbb futtasd a Mesh/DSM lépést.")
            build_orthophoto(self.prj, dsm, gsd=gsd, out_path=out,
                             log=self.log, progress=self._progress_cb)
        self._run_task("Orthofotó", task,
                       on_done=lambda: self._show_geotiff(out, "Orthofotó"))

    def mi_view_ortho(self):
        if self._need_project():
            return
        self._show_geotiff(os.path.join(self.prj.path, "ortho.tif"), "Orthofotó")

    def mi_view_dsm(self):
        if self._need_project():
            return
        path = os.path.join(self.prj.path, "dsm.tif")
        if not os.path.exists(path):
            messagebox.showinfo(APP_NAME, "Még nincs DSM.")
            return
        import rasterio
        with rasterio.open(path) as src:
            z = src.read(1)
        z = np.where(z == -9999.0, np.nan, z)
        zn = (z - np.nanmin(z)) / max(np.nanmax(z) - np.nanmin(z), 1e-9)
        import cv2
        cm = cv2.applyColorMap((np.nan_to_num(zn) * 255).astype(np.uint8),
                               cv2.COLORMAP_TURBO)[..., ::-1]
        self.current_photo = None
        self.marking_mode = False
        self.viewer_label.config(text="DSM (magasság-színezés)")
        self.viewer.set_image(Image.fromarray(cm))
        self.viewer.set_markers([])

    def _show_geotiff(self, path, label):
        if not os.path.exists(path):
            messagebox.showinfo(APP_NAME, f"Még nincs {label.lower()}.")
            return
        img = Image.open(path)
        img.load()
        if img.mode == "RGBA":
            img = img.convert("RGB")
        self.current_photo = None
        self.marking_mode = False
        self.viewer_label.config(text=f"{label}: {path}")
        self.viewer.set_image(img)
        self.viewer.set_markers([])

    def _on_close(self):
        if self.prj:
            try:
                self.prj.save()
            except Exception:
                pass
        self.destroy()


def _ask_choice(parent, title, prompt, choices, default):
    """Egyszerű választó-dialógus."""
    win = tk.Toplevel(parent)
    win.title(title)
    win.transient(parent)
    win.grab_set()
    ttk.Label(win, text=prompt).pack(padx=12, pady=(12, 4))
    var = tk.StringVar(value=default)
    for c in choices:
        ttk.Radiobutton(win, text=c, variable=var, value=c).pack(anchor="w",
                                                                 padx=20)
    result = {"v": None}

    def ok():
        result["v"] = var.get()
        win.destroy()

    ttk.Button(win, text="OK", command=ok).pack(pady=10)
    win.bind("<Return>", lambda e: ok())
    parent.wait_window(win)
    return result["v"]


def run_gui(project_path=None):
    app = App(project_path)
    app.mainloop()
