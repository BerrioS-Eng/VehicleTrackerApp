"""
core/preprocessor.py
Responsibility: Color space transformations on frames to facilitate subsequent segmentation/detection.
"""

import cv2
import numpy as np

class Preprocessor:
    """Frame preprocessing: color space conversions."""

    # Default range for isolating objects in HSV
    # (can be adjusted from config or the UI later)
    #Todo: Adjust range from the configuration or UI
    DEFAULT_HSV_LOWER = np.array([0, 50, 50])
    DEFAULT_HSV_UPPER = np.array([180, 255, 255])

    def __init__(self):
        self.hsv_lower = self.DEFAULT_HSV_LOWER.copy()
        self.hsv_upper = self.DEFAULT_HSV_UPPER.copy()

    def to_grayscale(self, frame):
        """
        Convierte un frame BGR a escala de grises.
        Útil para detección por bordes, gradientes y umbrales simples.
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def to_hsv(self, frame):
        """
        Convierte un frame BGR al espacio HSV completo.
        Retorna la imagen HSV de 3 canales.
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    def hsv_mask(self, frame):
        """
        Genera una máscara binaria a partir del rango HSV configurado.
        Permite aislar regiones por color/saturación/brillo.
        """
        hsv = self.to_hsv(frame)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        return mask
    
    def hsv_filtered(self, frame):
        """
        Aplica la máscara HSV sobre el frame original.
        Retorna solo las regiones que caen dentro del rango HSV.
        """
        mask = self.hsv_mask(frame)
        return cv2.bitwise_and(frame, frame, mask=mask)
    
    def combined(self, frame):
        """
        Estrategia combinada: aplica filtro HSV para aislar regiones
        de interés y luego convierte a escala de grises.
        Combina las ventajas de ambos enfoques.
        """
        filtered = self.hsv_filtered(frame)
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        return gray

    def set_hsv_range(self, lower, upper):
        """
        Permite ajustar el rango HSV dinámicamente.
        lower/upper: tuplas o arrays de 3 valores (H, S, V).
        """
        self.hsv_lower = np.array(lower)
        self.hsv_upper = np.array(upper)



