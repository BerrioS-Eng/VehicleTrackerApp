"""
main_window.py
Responsabilidad: construir y gestionar la ventana principal.
Layout 2x2 con ROI interactiva y filtro de color.
"""

import tkinter as tk
from tkinter import filedialog

import cv2
from PIL import Image, ImageTk

from config import settings
from core.video_handler import VideoHandler
from core.preprocessor import Preprocessor
from core.detector import Detector
from utils.drawing import draw_detections


class MainWindow:
    """Ventana principal de la aplicación."""

    COLOR_MODES = ["HSV", "Máscara HSV", "Filtrado HSV", "Combinado"]
    MORPH_MODES = ["Erosión", "Dilatación", "Apertura", "Cierre"]

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(settings.WINDOW_TITLE)
        self.root.geometry(
            f"{settings.WINDOW_WIDTH}x{settings.WINDOW_HEIGHT}"
        )
        self.root.resizable(True, True)

        # ── Módulos core ──
        self.video_handler = VideoHandler()
        self.preprocessor = Preprocessor()
        self.detector = Detector(min_area=500)

        # Referencias de imágenes
        self._images = {"original": None, "morph": None, "color": None}
        self._after_id = None
        self._last_frame = None
        self._last_detections = None

        # Variables para selectores
        self._color_mode = tk.StringVar(value="HSV")
        self._morph_mode = tk.StringVar(value="Apertura")
        self._bg_method = tk.StringVar(value="MOG2")

        # Variables para trackbars morfológicos
        self._kernel_size = tk.IntVar(value=5)
        self._iterations = tk.IntVar(value=1)

        # Variables para filtro de color
        self._color_filter_on = tk.BooleanVar(value=False)

        # Estado de dibujo de ROI
        self._drawing_roi = False
        self._roi_points_canvas = []  # Puntos en coords del canvas
        self._roi_temp_ids = []       # IDs de elementos temporales en canvas

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Construcción de la interfaz ────────────────────────

    def _build_ui(self):
        """Construye los widgets de la interfaz."""

        # --- Barra superior: controles ---
        toolbar = tk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.btn_open = tk.Button(
            toolbar, text="Abrir Video", command=self._open_video
        )
        self.btn_open.pack(side=tk.LEFT, padx=3)

        self.btn_play = tk.Button(
            toolbar, text="▶ Play", command=self._toggle_play,
            state=tk.DISABLED
        )
        self.btn_play.pack(side=tk.LEFT, padx=3)

        self.lbl_info = tk.Label(toolbar, text="Sin video cargado")
        self.lbl_info.pack(side=tk.LEFT, padx=10)

        tk.Label(toolbar, text="│  BG:").pack(side=tk.LEFT, padx=(15, 3))
        self.bg_selector = tk.OptionMenu(
            toolbar, self._bg_method,
            *Preprocessor.BG_METHODS,
            command=self._on_bg_method_change,
        )
        self.bg_selector.config(state=tk.DISABLED)
        self.bg_selector.pack(side=tk.LEFT, padx=3)

        self.lbl_detections = tk.Label(toolbar, text="")
        self.lbl_detections.pack(side=tk.RIGHT, padx=10)

        # --- Barra de herramientas: ROI y Color ---
        toolbar2 = tk.Frame(self.root)
        toolbar2.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        self.btn_roi = tk.Button(
            toolbar2, text="Dibujar ROI",
            command=self._start_roi_drawing, state=tk.DISABLED
        )
        self.btn_roi.pack(side=tk.LEFT, padx=3)

        self.btn_roi_clear = tk.Button(
            toolbar2, text="Limpiar ROI",
            command=self._clear_roi, state=tk.DISABLED
        )
        self.btn_roi_clear.pack(side=tk.LEFT, padx=3)

        self.lbl_roi_status = tk.Label(toolbar2, text="ROI: No definida", fg="gray")
        self.lbl_roi_status.pack(side=tk.LEFT, padx=10)

        tk.Label(toolbar2, text="│").pack(side=tk.LEFT, padx=5)

        self.chk_color = tk.Checkbutton(
            toolbar2, text="Filtro Rojo",
            variable=self._color_filter_on,
            command=self._on_color_filter_toggle,
            state=tk.DISABLED,
        )
        self.chk_color.pack(side=tk.LEFT, padx=3)

        # --- Grilla 2x2 ---
        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.grid_frame.rowconfigure(0, weight=1)
        self.grid_frame.rowconfigure(1, weight=1)
        self.grid_frame.columnconfigure(0, weight=1)
        self.grid_frame.columnconfigure(1, weight=1)

        self.panel_original = self._build_panel(
            self.grid_frame, row=0, col=0, title="Original + Detección"
        )
        self.panel_morph = self._build_morph_panel(
            self.grid_frame, row=0, col=1
        )
        self.panel_color = self._build_panel(
            self.grid_frame, row=1, col=0,
            title_var=self._color_mode, modes=self.COLOR_MODES
        )
        self.panel_graphs = self._build_panel(
            self.grid_frame, row=1, col=1, title="Gráficas (próximamente)"
        )

        # --- Barra inferior ---
        bottom = tk.Frame(self.root)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.lbl_frame = tk.Label(bottom, text="Frame: 0 / 0")
        self.lbl_frame.pack(side=tk.LEFT)

    def _build_panel(self, parent, row, col, title=None,
                     title_var=None, modes=None):
        frame = tk.Frame(parent, bd=1, relief=tk.SUNKEN)
        frame.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        panel = {}

        header = tk.Frame(frame)
        header.grid(row=0, column=0, sticky="ew")

        if title_var and modes:
            selector = tk.OptionMenu(
                header, title_var, *modes,
                command=lambda _: self._refresh_panels()
            )
            selector.config(state=tk.DISABLED)
            selector.pack(side=tk.LEFT, padx=3, pady=2)
            panel["selector"] = selector
        else:
            lbl = tk.Label(header, text=title, font=("Helvetica", 10, "bold"))
            lbl.pack(side=tk.LEFT, padx=5, pady=2)

        canvas = tk.Canvas(frame, bg="#1e1e1e", highlightthickness=0)
        canvas.grid(row=1, column=0, sticky="nsew")
        panel["canvas"] = canvas

        return panel

    def _build_morph_panel(self, parent, row, col):
        frame = tk.Frame(parent, bd=1, relief=tk.SUNKEN)
        frame.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        panel = {}

        header = tk.Frame(frame)
        header.grid(row=0, column=0, sticky="ew")

        selector = tk.OptionMenu(
            header, self._morph_mode, *self.MORPH_MODES,
            command=lambda _: self._refresh_panels()
        )
        selector.config(state=tk.DISABLED)
        selector.pack(side=tk.LEFT, padx=3, pady=2)
        panel["selector"] = selector

        tk.Label(header, text="Kernel:").pack(side=tk.LEFT, padx=(10, 2))
        self.scale_kernel = tk.Scale(
            header, from_=3, to=31, orient=tk.HORIZONTAL,
            variable=self._kernel_size, resolution=2,
            length=100, sliderlength=15,
            command=lambda _: self._on_morph_params_change(),
            state=tk.DISABLED,
        )
        self.scale_kernel.pack(side=tk.LEFT, padx=2)

        tk.Label(header, text="Iter:").pack(side=tk.LEFT, padx=(10, 2))
        self.scale_iter = tk.Scale(
            header, from_=1, to=10, orient=tk.HORIZONTAL,
            variable=self._iterations, resolution=1,
            length=100, sliderlength=15,
            command=lambda _: self._on_morph_params_change(),
            state=tk.DISABLED,
        )
        self.scale_iter.pack(side=tk.LEFT, padx=2)

        canvas = tk.Canvas(frame, bg="#1e1e1e", highlightthickness=0)
        canvas.grid(row=1, column=0, sticky="nsew")
        panel["canvas"] = canvas

        return panel

    # ── Acciones ───────────────────────────────────────────

    def _open_video(self):
        filetypes = [
            ("Videos", " ".join(f"*{ext}" for ext in settings.SUPPORTED_FORMATS)),
            ("Todos los archivos", "*.*"),
        ]
        path = filedialog.askopenfilename(
            title="Seleccionar video", filetypes=filetypes,
        )
        if not path:
            return

        if self.video_handler.open(path):
            info = self.video_handler.get_info()
            self.lbl_info.config(
                text=(
                    f"{info['width']}x{info['height']}  |  "
                    f"{info['fps']:.1f} FPS  |  "
                    f"{info['duration_sec']:.1f}s"
                )
            )
            self.btn_play.config(state=tk.NORMAL)
            self.preprocessor.reset_bg_subtractors()
            self.preprocessor.clear_roi()
            self.lbl_roi_status.config(text="ROI: No definida", fg="gray")
            self._enable_selectors()
            self._show_first_frame()
        else:
            self.lbl_info.config(text="Error al abrir el video")

    def _enable_selectors(self):
        if "selector" in self.panel_color:
            self.panel_color["selector"].config(state=tk.NORMAL)
        if "selector" in self.panel_morph:
            self.panel_morph["selector"].config(state=tk.NORMAL)
        self.bg_selector.config(state=tk.NORMAL)
        self.scale_kernel.config(state=tk.NORMAL)
        self.scale_iter.config(state=tk.NORMAL)
        self.btn_roi.config(state=tk.NORMAL)
        self.btn_roi_clear.config(state=tk.NORMAL)
        self.chk_color.config(state=tk.NORMAL)

    def _show_first_frame(self):
        ret, frame = self.video_handler.read_frame()
        if ret:
            self._last_frame = frame
            self._render_all_panels(frame)
            self._update_frame_label()

    def _toggle_play(self):
        if self.video_handler.is_playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        self.video_handler.is_playing = True
        self.btn_play.config(text="⏸ Pausa")
        self._play_loop()

    def _pause(self):
        self.video_handler.is_playing = False
        self.btn_play.config(text="▶ Play")
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _play_loop(self):
        if not self.video_handler.is_playing:
            return

        ret, frame = self.video_handler.read_frame()
        if ret:
            self._last_frame = frame
            self._render_all_panels(frame)
            self._update_frame_label()
            delay = int(1000 / self.video_handler.fps)
            self._after_id = self.root.after(delay, self._play_loop)
        else:
            self._pause()

    def _on_bg_method_change(self, _):
        self.preprocessor.set_bg_method(self._bg_method.get())
        self._refresh_panels()

    def _on_morph_params_change(self):
        size = self._kernel_size.get()
        iterations = self._iterations.get()
        self.preprocessor.set_kernel(size=size)

        mode = self._morph_mode.get()
        if mode == "Erosión":
            self.preprocessor.erosion_iter = iterations
        elif mode == "Dilatación":
            self.preprocessor.dilation_iter = iterations
        elif mode == "Apertura":
            self.preprocessor.opening_iter = iterations
        elif mode == "Cierre":
            self.preprocessor.closing_iter = iterations

        self._refresh_panels()

    def _on_color_filter_toggle(self):
        self.preprocessor.color_filter_enabled = self._color_filter_on.get()
        self._refresh_panels()

    def _refresh_panels(self):
        if self._last_frame is not None:
            self._render_all_panels(self._last_frame)

    # ── ROI interactiva ────────────────────────────────────

    def _start_roi_drawing(self):
        """Activa el modo de dibujo de ROI sobre el panel original."""
        if self._last_frame is None:
            return

        self._pause()
        self._drawing_roi = True
        self._roi_points_canvas = []
        self._roi_temp_ids = []

        self.btn_roi.config(text="Finalizar ROI", command=self._finish_roi)
        self.lbl_roi_status.config(text="ROI: Haz clic para definir puntos...", fg="orange")

        canvas = self.panel_original["canvas"]
        canvas.bind("<Button-1>", self._on_roi_click)

    def _on_roi_click(self, event):
        """Registra un punto de la ROI al hacer clic en el canvas."""
        if not self._drawing_roi:
            return

        canvas = self.panel_original["canvas"]

        # Dibujar punto
        r = 4
        point_id = canvas.create_oval(
            event.x - r, event.y - r, event.x + r, event.y + r,
            fill="lime", outline="lime"
        )
        self._roi_temp_ids.append(point_id)

        # Dibujar línea al punto anterior
        if self._roi_points_canvas:
            px, py = self._roi_points_canvas[-1]
            line_id = canvas.create_line(
                px, py, event.x, event.y, fill="lime", width=2
            )
            self._roi_temp_ids.append(line_id)

        self._roi_points_canvas.append((event.x, event.y))

    def _finish_roi(self):
        """Finaliza el dibujo de la ROI y la aplica."""
        canvas = self.panel_original["canvas"]
        canvas.unbind("<Button-1>")

        # Limpiar elementos temporales
        for item_id in self._roi_temp_ids:
            canvas.delete(item_id)
        self._roi_temp_ids = []

        if len(self._roi_points_canvas) < 3:
            self.lbl_roi_status.config(text="ROI: Mínimo 3 puntos", fg="red")
            self._drawing_roi = False
            self.btn_roi.config(text="Dibujar ROI", command=self._start_roi_drawing)
            return

        # Convertir coords del canvas a coords del frame original
        frame_points = self._canvas_to_frame_coords(self._roi_points_canvas)

        self.preprocessor.set_roi(frame_points, self._last_frame.shape)
        self._drawing_roi = False

        n = len(frame_points)
        self.btn_roi.config(text="Dibujar ROI", command=self._start_roi_drawing)
        self.lbl_roi_status.config(text=f"ROI: Activa ({n} puntos)", fg="green")

        self._refresh_panels()

    def _clear_roi(self):
        """Elimina la ROI definida."""
        self.preprocessor.clear_roi()
        self.lbl_roi_status.config(text="ROI: No definida", fg="gray")
        self._refresh_panels()

    def _canvas_to_frame_coords(self, canvas_points):
        """
        Convierte puntos del canvas a coordenadas del frame original.
        Tiene en cuenta el escalado y centrado de la imagen.
        """
        canvas = self.panel_original["canvas"]
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()

        img_h, img_w = self._last_frame.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        # Offset del centrado
        offset_x = (canvas_w - new_w) / 2
        offset_y = (canvas_h - new_h) / 2

        frame_points = []
        for cx, cy in canvas_points:
            fx = int((cx - offset_x) / scale)
            fy = int((cy - offset_y) / scale)
            # Clamp a los límites del frame
            fx = max(0, min(fx, img_w - 1))
            fy = max(0, min(fy, img_h - 1))
            frame_points.append((fx, fy))

        return frame_points

    # ── Procesamiento central ──────────────────────────────

    def _process_frame(self, frame):
        mask = self.preprocessor.apply_full_pipeline(frame)
        self._last_detections = self.detector.detect(mask)
        return self._last_detections

    # ── Modos de vista ─────────────────────────────────────

    def _apply_color_mode(self, frame, mode):
        pp = self.preprocessor

        if mode == "HSV":
            result = pp.to_hsv(frame)
            return cv2.cvtColor(result, cv2.COLOR_HSV2RGB)
        elif mode == "Máscara HSV":
            result = pp.hsv_mask(frame)
            return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        elif mode == "Filtrado HSV":
            result = pp.hsv_filtered(frame)
            return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        elif mode == "Combinado":
            result = pp.combined(frame)
            return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _apply_morph_mode(self, frame, mode):
        pp = self.preprocessor
        mask = pp.apply_bg_subtraction_clean(frame)

        # Aplicar ROI también en la vista de morfología
        mask = pp.apply_roi(mask)

        if mode == "Erosión":
            result = pp.apply_erosion(mask)
        elif mode == "Dilatación":
            result = pp.apply_dilation(mask)
        elif mode == "Apertura":
            result = pp.apply_opening(mask)
        elif mode == "Cierre":
            result = pp.apply_closing(mask)
        else:
            result = mask

        return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    # ── Renderizado ────────────────────────────────────────

    def _render_all_panels(self, frame):
        self._process_frame(frame)
        n = len(self._last_detections["contours"])
        self.lbl_detections.config(text=f"Detectados: {n}")

        # Sup-Izq: Original + ROI overlay + detección
        annotated = frame.copy()
        annotated = self.preprocessor.get_roi_overlay(annotated)
        if self._last_detections:
            draw_detections(annotated, self._last_detections)
        original_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        self._draw_on_canvas(
            self.panel_original["canvas"], original_rgb, "original"
        )

        # Sup-Der: Morfología
        morph_rgb = self._apply_morph_mode(frame, self._morph_mode.get())
        self._draw_on_canvas(
            self.panel_morph["canvas"], morph_rgb, "morph"
        )

        # Inf-Izq: Espacio de color
        color_rgb = self._apply_color_mode(frame, self._color_mode.get())
        self._draw_on_canvas(
            self.panel_color["canvas"], color_rgb, "color"
        )

    def _draw_on_canvas(self, canvas, frame_rgb, image_key):
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()

        if canvas_w < 10 or canvas_h < 10:
            canvas_w = settings.WINDOW_WIDTH // 2
            canvas_h = (settings.WINDOW_HEIGHT - 80) // 2

        img_h, img_w = frame_rgb.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        pil_image = Image.fromarray(frame_resized)
        self._images[image_key] = ImageTk.PhotoImage(pil_image)

        canvas.delete("all")
        canvas.create_image(
            canvas_w // 2, canvas_h // 2,
            anchor=tk.CENTER, image=self._images[image_key],
        )

    def _update_frame_label(self):
        self.lbl_frame.config(
            text=(
                f"Frame: {self.video_handler.current_frame} / "
                f"{self.video_handler.total_frames}"
            )
        )

    # ── Ciclo de vida ──────────────────────────────────────

    def _on_close(self):
        self._pause()
        self.video_handler.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()