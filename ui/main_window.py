"""
main_window.py
Responsabilidad: construir y gestionar la ventana principal.
Layout 2x2: Original | Preprocesado 1
             Preprocesado 2 | Gráficas
"""

import tkinter as tk
from tkinter import filedialog

import cv2
from PIL import Image, ImageTk

from config import settings
from core.video_handler import VideoHandler
from core.preprocessor import Preprocessor


class MainWindow:
    """Ventana principal de la aplicación."""

    PREPROCESS_MODES = ["HSV", "Máscara HSV", "Filtrado HSV", "Combinado"]

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

        # Referencias de imágenes (evita recolección de basura)
        self._images = {"original": None, "pre1": None, "pre2": None}
        self._after_id = None
        self._last_frame = None

        # Variables para los selectores de modo
        self._mode1 = tk.StringVar(value="HSV")
        self._mode2 = tk.StringVar(value="Máscara HSV")

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

        # --- Grilla 2x2 ---
        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.grid_frame.rowconfigure(0, weight=1)
        self.grid_frame.rowconfigure(1, weight=1)
        self.grid_frame.columnconfigure(0, weight=1)
        self.grid_frame.columnconfigure(1, weight=1)

        # Panel superior izquierdo: Original
        self.panel_original = self._build_panel(
            self.grid_frame, row=0, col=0, title="Original"
        )

        # Panel superior derecho: Preprocesado 1
        self.panel_pre1 = self._build_panel(
            self.grid_frame, row=0, col=1,
            title_var=self._mode1, modes=self.PREPROCESS_MODES
        )

        # Panel inferior izquierdo: Preprocesado 2
        self.panel_pre2 = self._build_panel(
            self.grid_frame, row=1, col=0,
            title_var=self._mode2, modes=self.PREPROCESS_MODES
        )

        # Panel inferior derecho: Gráficas (placeholder)
        self.panel_graphs = self._build_panel(
            self.grid_frame, row=1, col=1, title="Gráficas (próximamente)"
        )

        # --- Barra inferior: progreso ---
        bottom = tk.Frame(self.root)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.lbl_frame = tk.Label(bottom, text="Frame: 0 / 0")
        self.lbl_frame.pack(side=tk.LEFT)

    def _build_panel(self, parent, row, col, title=None,
                     title_var=None, modes=None):
        """
        Construye un panel con etiqueta/selector arriba y canvas abajo.
        Retorna un dict con las referencias al canvas y al selector.
        """
        frame = tk.Frame(parent, bd=1, relief=tk.SUNKEN)
        frame.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        panel = {}

        # Cabecera del panel
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

        # Canvas
        canvas = tk.Canvas(frame, bg="#1e1e1e", highlightthickness=0)
        canvas.grid(row=1, column=0, sticky="nsew")
        panel["canvas"] = canvas

        return panel

    # ── Acciones ───────────────────────────────────────────

    def _open_video(self):
        """Abre un diálogo para seleccionar un archivo de video."""
        filetypes = [
            ("Videos", " ".join(f"*{ext}" for ext in settings.SUPPORTED_FORMATS)),
            ("Todos los archivos", "*.*"),
        ]
        path = filedialog.askopenfilename(
            title="Seleccionar video",
            filetypes=filetypes,
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
            self._enable_selectors()
            self._show_first_frame()
        else:
            self.lbl_info.config(text="Error al abrir el video")

    def _enable_selectors(self):
        """Habilita los selectores de modo al cargar video."""
        for panel in (self.panel_pre1, self.panel_pre2):
            if "selector" in panel:
                panel["selector"].config(state=tk.NORMAL)

    def _disable_selectors(self):
        """Deshabilita los selectores cuando no hay video."""
        for panel in (self.panel_pre1, self.panel_pre2):
            if "selector" in panel:
                panel["selector"].config(state=tk.DISABLED)

    def _show_first_frame(self):
        """Lee y muestra el primer frame como vista previa."""
        ret, frame = self.video_handler.read_frame()
        if ret:
            self._last_frame = frame
            self._render_all_panels(frame)
            self._update_frame_label()

    def _toggle_play(self):
        """Alterna entre reproducir y pausar."""
        if self.video_handler.is_playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        """Inicia la reproducción del video."""
        self.video_handler.is_playing = True
        self.btn_play.config(text="⏸ Pausa")
        self._play_loop()

    def _pause(self):
        """Pausa la reproducción."""
        self.video_handler.is_playing = False
        self.btn_play.config(text="▶ Play")
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _play_loop(self):
        """Loop de reproducción: lee y muestra frames al ritmo del FPS."""
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

    def _refresh_panels(self):
        """Re-renderiza los paneles con el último frame (para cambio de modo en pausa)."""
        if self._last_frame is not None:
            self._render_all_panels(self._last_frame)

    # ── Preprocesamiento ───────────────────────────────────

    def _apply_mode(self, frame, mode):
        """
        Aplica la transformación del modo indicado.
        Retorna siempre una imagen RGB lista para mostrar.
        """
        if mode == "HSV":
            return self.preprocessor.to_hsv(frame)
        
        elif mode == "Máscara HSV":
            mask = self.preprocessor.hsv_mask(frame)
            return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        elif mode == "Filtrado HSV":
            filtered = self.preprocessor.hsv_filtered(frame)
            return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)

        elif mode == "Combinado":
            combined = self.preprocessor.combined(frame)
            return cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── Renderizado ────────────────────────────────────────

    def _render_all_panels(self, frame):
        """Renderiza los 3 paneles activos con el frame dado."""
        # Original
        original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._draw_on_canvas(
            self.panel_original["canvas"], original_rgb, "original"
        )

        # Preprocesado 1
        pre1_rgb = self._apply_mode(frame, self._mode1.get())
        self._draw_on_canvas(
            self.panel_pre1["canvas"], pre1_rgb, "pre1"
        )

        # Preprocesado 2
        pre2_rgb = self._apply_mode(frame, self._mode2.get())
        self._draw_on_canvas(
            self.panel_pre2["canvas"], pre2_rgb, "pre2"
        )

    def _draw_on_canvas(self, canvas, frame_rgb, image_key):
        """
        Dibuja un frame RGB en un canvas, escalado proporcionalmente.
        image_key: clave en self._images para mantener la referencia.
        """
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
            anchor=tk.CENTER,
            image=self._images[image_key],
        )

    def _update_frame_label(self):
        """Actualiza el indicador de frame actual."""
        self.lbl_frame.config(
            text=(
                f"Frame: {self.video_handler.current_frame} / "
                f"{self.video_handler.total_frames}"
            )
        )

    # ── Ciclo de vida ──────────────────────────────────────

    def _on_close(self):
        """Limpia recursos antes de cerrar."""
        self._pause()
        self.video_handler.release()
        self.root.destroy()

    def run(self):
        """Inicia el loop principal de Tkinter."""
        self.root.mainloop()