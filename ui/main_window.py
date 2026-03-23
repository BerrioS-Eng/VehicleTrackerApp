"""
main_window.py
Responsibility: build and manage the main app window (Tkinter).
Display the video frame by frame.
"""
import tkinter as tk
from tkinter import filedialog

import cv2
from PIL import Image, ImageTk

from config import settings
from core.video_handler import VideoHandler
from core.preprocessor import Preprocessor


class MainWindow:
    """Main application window."""

    VIEW_MODES = ["Original", "Grises", "HSV", "Máscara HSV", "Filtrado HSV", "Combinado"]

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(settings.WINDOW_TITLE)
        self.root.geometry(
            f"{settings.WINDOW_WIDTH}x{settings.WINDOW_HEIGHT}"
        )
        self.root.resizable(True, True)


        # --- Core Module ---
        self.video_handler = VideoHandler()
        self.preprocessor = Preprocessor()

        # Reference to avoid garbage collection from the image.
        self.current_image = None

        # ID of callback pending from after()
        self.after_id = None

        # Last frame read (to change mode without advancing the video)
        self._last_frame = None

        # Variable for view mode
        self._view_mode = tk.StringVar(value="Original")

        self.build_ui()

        # Clear resources when closing window.
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Interface construction ────────────────────────
    def build_ui(self):
        """Build the interfaces widgets."""

        # --- Top bar: controls ---
        toolbar = tk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.btn_open = tk.Button(
            toolbar, text="Open video", command=self._open_video
        )
        self.btn_open.pack(side=tk.LEFT, padx=3)

        self.btn_play = tk.Button(
            toolbar, text="▶ Play", command=self._toggle_play, state=tk.DISABLED
        )
        self.btn_play.pack(side=tk.LEFT, padx=3)

        self.lbl_info = tk.Label(toolbar, text="No video loaded")
        self.lbl_info.pack(side=tk.LEFT, padx=10)

        # --- View mode selector ---
        view_frame = tk.Frame(self.root)
        view_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(view_frame, text="Vista:").pack(side=tk.LEFT, padx=(0, 5))

        for mode in self.VIEW_MODES:
            tk.Radiobutton(
                view_frame, 
                text=mode, 
                variable=self._view_mode,
                value=mode,
                command=self._on_view_mode_change,
            ).pack(side=tk.LEFT, padx=2)

        # --- Central canvas: video area ---
        self.canvas = tk.Canvas(
            self.root, bg="#1e1e1e", highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # --- Bottom bar: progress ---
        bottom = tk.Frame(self.root)
        bottom.pack(
            side=tk.BOTTOM, fill=tk.X, padx=5, pady=5
        )

        self.lbl_frame = tk.Label(bottom, text="Frame: 0 / 0")
        self.lbl_frame.pack(side=tk.LEFT)

    # ── Actions ──────────────────────── 
    def _open_video(self):
        """Opens a dialog to select a video file."""
        filetypes = [
            ("Videos", " ".join(f"*{ext}" for ext in settings.SUPPORTED_FORMATS)),
            ("Todos los archivos", "*.*"),
        ]
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=filetypes,
        )
        if not path:
            return
        
        if self.video_handler.open(path):
            info = self.video_handler.get_info()
            self.lbl_info.config(
                text= (
                    f"{info['width']}x{info['height']}  |  "
                    f"{info['fps']:.1f} FPS  |  "
                    f"{info['duration_sec']:.1f}s"
                )
            )
            self.btn_play.config(state=tk.NORMAL)
            self._show_first_frame()
        else:
            self.lbl_info.config(text="Error opening video.")
    
    def _show_first_frame(self):
        """Read and display the first frame as a preview."""
        ret, frame = self.video_handler.read_frame()
        if ret:
            self._display_frame(frame)
            self._update_frame_label()
    
    def _toggle_play(self):
        """Toogle between play and pause."""
        if self.video_handler.is_playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        """Video playbacks begins."""
        self.video_handler.is_playing = True
        self.btn_play.config(text="⏸ Pausa")
        self._play_loop()

    def _pause(self):
        """Pause playback."""
        self.video_handler.is_playing = False
        self.btn_play.config( text="▶ Play")
        if self.after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
    
    def _play_loop(self):
        """Playback loop: reads and displays frames at the FPS rate."""
        #Todo: Consider adding a playback loop
        if not self.video_handler.is_playing:
            return

        ret, frame = self.video_handler.read_frame()

        if ret: 
            self._display_frame(frame)
            self._update_frame_label()
            delay = int(1000 / self.video_handler.fps)
            self._after_id = self.root.after(delay, self._play_loop)
        else:
            # end video.
            self._pause()
    
    def _on_view_mode_change(self):
        """Callback when the user changes the view mode."""
        if self._last_frame is not None:
            self._display_frame(self._last_frame)

    # ── Preprocessing ────────────────────────────────────────
    def _apply_preprocessing(self, frame):
        """
        Applies the transformation corresponding to the selected view mode.
        Always returns a display-ready image (RGB).
        """
        mode = self._view_mode.get()

        if mode == "Original":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif mode == "Grises":
            gray = self.preprocessor.to_grayscale(frame)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        elif mode == "HSV":
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
        
    # ── Rendering ────────────────────────────────────────
    def _display_frame(self, frame):
        """
        Convert an OpenCV frame (BGR/numpy) to a Tkinter image.
        And draw it on the canvas, scaled to the available size.
        """
        # Apply preprocessing according to the selected mode
        frame_rgb = self._apply_preprocessing(frame)

        # Scale to the canvas size while maintaining aspect ratio.
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10:
            canvas_w = settings.WINDOW_WIDTH
            canvas_h = settings.WINDOW_HEIGHT - 80

        img_h, img_w = frame_rgb.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))

        # Numpy -> PIL -> ImageTk
        pil_image = Image.fromarray(frame_resized)
        self.current_image = ImageTk.PhotoImage(pil_image)

        # Draw centered on the canvas
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_w // 2, canvas_h // 2,
            anchor= tk.CENTER, 
            image= self.current_image,
        )

    def _update_frame_label(self):
        """Update the current frame indicator."""
        self.lbl_frame.config(
            text=(
                f"Frame: {self.video_handler.current_frame} / "
                f"{self.video_handler.total_frames}"
            )
        )

    # ── Lifecycle ──────────────────────────────────────
    def _on_close(self):
        """Clean up resources before closing."""
        self._pause()
        self.video_handler.release()
        self.root.destroy()

    def run(self):
        """Start the main Tkinter loop."""
        self.root.mainloop()



