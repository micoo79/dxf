"""Nagyítható-mozgatható képnéző vászon jelölő-elhelyezéssel.

Nagy képeknél csak a látható kivágást rendereli az aktuális nagyításon,
így a navigáció nagy fotókon is gyors marad.
"""

import tkinter as tk

from PIL import Image, ImageTk

Image.MAX_IMAGE_PIXELS = None  # nagy orthofotók megnyitásához


class ImageView(tk.Canvas):
    """Canvas, amely egy PIL-képet jelenít meg zoommal és pannal.

    Koordináták: "kép-pixel" = a betöltött kép teljes felbontású pixele.
    on_click(u, v) hívódik bal kattintásra (kép-koordinátákkal),
    on_right_click(u, v) jobb kattintásra.
    """

    def __init__(self, master, **kw):
        super().__init__(master, bg="#202020", highlightthickness=0, **kw)
        self.img = None          # PIL Image
        self.zoom = 1.0
        self.off_x = 0.0         # a vászon (0,0)-jának kép-koordinátája
        self.off_y = 0.0
        self.markers = []        # (u, v, szín, felirat)
        self.on_click = None
        self.on_right_click = None
        self._drag = None
        self._photo = None       # ImageTk referencia (GC ellen)
        self._moved = False

        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<ButtonPress-3>", self._right)
        self.bind("<MouseWheel>", self._wheel)          # Windows/mac
        self.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1.25))
        self.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 0.8))

    # ------------------------------------------------------------- publikus

    def set_image(self, pil_img, keep_view=False):
        self.img = pil_img
        if not keep_view:
            self.fit()
        else:
            self.redraw()

    def set_markers(self, markers):
        """markers: list of (u, v, szín, felirat)"""
        self.markers = list(markers)
        self.redraw()

    def fit(self):
        if self.img is None:
            return
        cw = max(self.winfo_width(), 1)
        ch = max(self.winfo_height(), 1)
        self.zoom = min(cw / self.img.width, ch / self.img.height)
        self.zoom = min(self.zoom, 4.0)
        self.off_x = (self.img.width - cw / self.zoom) / 2
        self.off_y = (self.img.height - ch / self.zoom) / 2
        self.redraw()

    def center_on(self, u, v, zoom=None):
        if self.img is None:
            return
        if zoom:
            self.zoom = zoom
        cw = max(self.winfo_width(), 1)
        ch = max(self.winfo_height(), 1)
        self.off_x = u - cw / self.zoom / 2
        self.off_y = v - ch / self.zoom / 2
        self.redraw()

    # ---------------------------------------------------------- események

    def _canvas_to_img(self, x, y):
        return self.off_x + x / self.zoom, self.off_y + y / self.zoom

    def _press(self, e):
        self._drag = (e.x, e.y, self.off_x, self.off_y)
        self._moved = False

    def _motion(self, e):
        if self._drag is None:
            return
        x0, y0, ox, oy = self._drag
        dx, dy = e.x - x0, e.y - y0
        if abs(dx) + abs(dy) > 3:
            self._moved = True
        self.off_x = ox - dx / self.zoom
        self.off_y = oy - dy / self.zoom
        self.redraw()

    def _release(self, e):
        if self._drag is not None and not self._moved and self.on_click:
            u, v = self._canvas_to_img(e.x, e.y)
            if self.img is not None and 0 <= u < self.img.width and 0 <= v < self.img.height:
                self.on_click(u, v)
        self._drag = None

    def _right(self, e):
        if self.on_right_click:
            u, v = self._canvas_to_img(e.x, e.y)
            self.on_right_click(u, v)

    def _wheel(self, e):
        self._zoom_at(e.x, e.y, 1.25 if e.delta > 0 else 0.8)

    def _zoom_at(self, cx, cy, factor):
        if self.img is None:
            return
        u, v = self._canvas_to_img(cx, cy)
        self.zoom = max(0.02, min(self.zoom * factor, 32.0))
        self.off_x = u - cx / self.zoom
        self.off_y = v - cy / self.zoom
        self.redraw()

    # ------------------------------------------------------------ rajzolás

    def redraw(self):
        self.delete("all")
        if self.img is None:
            return
        cw = max(self.winfo_width(), 1)
        ch = max(self.winfo_height(), 1)
        # látható kivágás kép-koordinátákban
        x0 = max(0.0, self.off_x)
        y0 = max(0.0, self.off_y)
        x1 = min(float(self.img.width), self.off_x + cw / self.zoom)
        y1 = min(float(self.img.height), self.off_y + ch / self.zoom)
        if x1 <= x0 or y1 <= y0:
            return
        crop = self.img.crop((int(x0), int(y0),
                              min(int(x1) + 1, self.img.width),
                              min(int(y1) + 1, self.img.height)))
        disp_w = max(1, int(crop.width * self.zoom))
        disp_h = max(1, int(crop.height * self.zoom))
        resample = Image.NEAREST if self.zoom >= 3 else Image.BILINEAR
        crop = crop.resize((disp_w, disp_h), resample)
        self._photo = ImageTk.PhotoImage(crop)
        px = (int(x0) - self.off_x) * self.zoom
        py = (int(y0) - self.off_y) * self.zoom
        self.create_image(px, py, image=self._photo, anchor="nw")
        # jelölők
        for (u, v, color, label) in self.markers:
            x = (u - self.off_x) * self.zoom
            y = (v - self.off_y) * self.zoom
            if -50 <= x <= cw + 50 and -50 <= y <= ch + 50:
                r = 12
                self.create_line(x - r, y, x + r, y, fill=color, width=2)
                self.create_line(x, y - r, x, y + r, fill=color, width=2)
                self.create_oval(x - r * 0.6, y - r * 0.6, x + r * 0.6,
                                 y + r * 0.6, outline=color, width=2)
                if label:
                    self.create_text(x + r + 4, y - r - 4, text=label,
                                     fill=color, anchor="w",
                                     font=("TkDefaultFont", 10, "bold"))
