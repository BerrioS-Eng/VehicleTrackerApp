"""
preprocessor.py
Responsabilidad: transformaciones de espacio de color,
sustracción de fondo, operaciones morfológicas,
ROI y filtrado por color para segmentación/detección.
"""

import cv2
import numpy as np


class Preprocessor:
    """Preprocesamiento de frames: color, BG, morfología, ROI."""

    DEFAULT_HSV_LOWER = np.array([0, 50, 50])
    DEFAULT_HSV_UPPER = np.array([180, 255, 255])

    # Rangos HSV para rojo más restrictivos en S y V
    # para rechazar sombras (baja saturación, bajo valor)
    RED_RANGES = [
        (np.array([0, 150, 80]), np.array([10, 255, 255])),
        (np.array([170, 150, 80]), np.array([180, 255, 255])),
    ]

    BG_METHODS = ["MOG2", "KNN"]

    def __init__(self):
        self.hsv_lower = self.DEFAULT_HSV_LOWER.copy()
        self.hsv_upper = self.DEFAULT_HSV_UPPER.copy()

        # Sustractores de fondo
        self._bg_subtractors = {
            "MOG2": cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=50, detectShadows=True,
            ),
            "KNN": cv2.createBackgroundSubtractorKNN(
                history=500, dist2Threshold=400.0, detectShadows=True,
            ),
        }
        self._active_bg_method = "MOG2"

        # Parámetros morfológicos
        self._kernel_size = 5
        self._kernel_shape = cv2.MORPH_ELLIPSE
        self._kernel = self._make_kernel()

        self.erosion_iter = 1
        self.dilation_iter = 1
        self.opening_iter = 1
        self.closing_iter = 2

        # Desenfoque previo a la sustracción de fondo
        self.blur_size = 5

        # ROI
        self._roi_points = None
        self._roi_mask = None

        # Filtrado por color
        self.color_filter_enabled = False
        self._color_ranges = self.RED_RANGES

    def _make_kernel(self):
        return cv2.getStructuringElement(
            self._kernel_shape, (self._kernel_size, self._kernel_size)
        )

    def set_kernel(self, size=5, shape="ellipse"):
        shapes = {
            "ellipse": cv2.MORPH_ELLIPSE,
            "rect": cv2.MORPH_RECT,
            "cross": cv2.MORPH_CROSS,
        }
        self._kernel_size = size
        self._kernel_shape = shapes.get(shape, cv2.MORPH_ELLIPSE)
        self._kernel = self._make_kernel()

    # ── ROI ────────────────────────────────────────────────

    def set_roi(self, points, frame_shape):
        self._roi_points = points
        h, w = frame_shape[:2]
        self._roi_mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(self._roi_mask, [pts], 255)

    def clear_roi(self):
        self._roi_points = None
        self._roi_mask = None

    def has_roi(self) -> bool:
        return self._roi_mask is not None

    def get_roi_points(self):
        return self._roi_points

    def apply_roi(self, mask):
        if self._roi_mask is not None:
            return cv2.bitwise_and(mask, self._roi_mask)
        return mask

    def get_roi_overlay(self, frame):
        if self._roi_points is None:
            return frame
        overlay = frame.copy()
        pts = np.array(self._roi_points, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
        return cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)

    # ── Filtrado por color ─────────────────────────────────

    def set_color_ranges(self, ranges):
        self._color_ranges = ranges

    def apply_color_filter(self, frame):
        """
        Genera máscara binaria para los rangos de color definidos.
        Incluye dilatación para cubrir bordes del objeto
        que podrían no ser exactamente rojos.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in self._color_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Dilatar la máscara de color para capturar el objeto completo,
        # no solo los píxeles estrictamente rojos
        color_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        combined_mask = cv2.dilate(combined_mask, color_kernel, iterations=2)

        return combined_mask

    # ── Espacio de color ───────────────────────────────────

    def to_grayscale(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def to_hsv(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def hsv_mask(self, frame):
        hsv = self.to_hsv(frame)
        return cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

    def hsv_filtered(self, frame):
        mask = self.hsv_mask(frame)
        return cv2.bitwise_and(frame, frame, mask=mask)

    def combined(self, frame):
        filtered = self.hsv_filtered(frame)
        return cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

    def set_hsv_range(self, lower, upper):
        self.hsv_lower = np.array(lower)
        self.hsv_upper = np.array(upper)

    # ── Sustracción de fondo ───────────────────────────────

    def set_bg_method(self, method: str):
        if method in self._bg_subtractors:
            self._active_bg_method = method

    def get_bg_method(self) -> str:
        return self._active_bg_method

    def apply_bg_subtraction(self, frame):
        """Aplica desenfoque + sustracción de fondo."""
        blurred = cv2.GaussianBlur(
            frame, (self.blur_size, self.blur_size), 0
        )
        subtractor = self._bg_subtractors[self._active_bg_method]
        return subtractor.apply(blurred)

    def apply_bg_subtraction_clean(self, frame):
        fg_mask = self.apply_bg_subtraction(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        return fg_mask

    def get_foreground(self, frame):
        mask = self.apply_full_pipeline(frame)
        return cv2.bitwise_and(frame, frame, mask=mask)

    def reset_bg_subtractors(self):
        self._bg_subtractors["MOG2"] = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True,
        )
        self._bg_subtractors["KNN"] = cv2.createBackgroundSubtractorKNN(
            history=500, dist2Threshold=400.0, detectShadows=True,
        )

    # ── Operaciones morfológicas ───────────────────────────

    def apply_erosion(self, mask):
        return cv2.erode(mask, self._kernel, iterations=self.erosion_iter)

    def apply_dilation(self, mask):
        return cv2.dilate(mask, self._kernel, iterations=self.dilation_iter)

    def apply_opening(self, mask):
        return cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, self._kernel, iterations=self.opening_iter
        )

    def apply_closing(self, mask):
        return cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, self._kernel, iterations=self.closing_iter
        )

    # ── Pipeline completo ──────────────────────────────────

    def apply_full_pipeline(self, frame):
        """
        Pipeline completo:
        1. Desenfoque + sustracción de fondo + eliminación de sombras
        2. Filtro de color (si habilitado)
        3. ROI (si definida)
        4. Apertura → Cierre
        """
        mask = self.apply_bg_subtraction_clean(frame)

        if self.color_filter_enabled:
            color_mask = self.apply_color_filter(frame)
            mask = cv2.bitwise_and(mask, color_mask)

        mask = self.apply_roi(mask)

        mask = self.apply_opening(mask)
        mask = self.apply_closing(mask)

        return mask