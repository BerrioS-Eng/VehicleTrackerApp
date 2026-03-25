"""
detector.py
Responsabilidad: detectar vehículos a partir de la máscara
procesada, encontrando y filtrando contornos.
"""

import cv2


class Detector:
    """Detecta vehículos mediante análisis de contornos."""

    def __init__(self, min_area=500, max_area=None,
                 min_aspect=0.3, max_aspect=3.5,
                 min_solidity=0.4):
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect    # ancho/alto mínimo
        self.max_aspect = max_aspect    # ancho/alto máximo
        self.min_solidity = min_solidity  # solidez mínima (área/convex hull)

    def find_contours(self, mask):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def filter_contours(self, contours):
        """
        Filtra contornos por múltiples criterios:
        - Área: descarta ruido (muy pequeños) y falsos (muy grandes)
        - Aspect ratio: descarta formas demasiado alargadas (sombras)
        - Solidez: descarta formas muy irregulares (fragmentos)
        """
        filtered = []
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Filtro por área
            if area < self.min_area:
                continue
            if self.max_area is not None and area > self.max_area:
                continue

            # Filtro por aspect ratio
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0:
                continue
            aspect = w / h
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue

            # Filtro por solidez (qué tan "lleno" es el contorno)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < self.min_solidity:
                continue

            filtered.append(cnt)
        return filtered

    def get_bounding_boxes(self, contours):
        return [cv2.boundingRect(cnt) for cnt in contours]

    def get_centroids(self, contours):
        centroids = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
        return centroids

    def detect(self, mask):
        all_contours = self.find_contours(mask)
        valid_contours = self.filter_contours(all_contours)
        bboxes = self.get_bounding_boxes(valid_contours)
        centroids = self.get_centroids(valid_contours)

        return {
            "contours": valid_contours,
            "bounding_boxes": bboxes,
            "centroids": centroids,
        }

    def set_area_range(self, min_area=None, max_area=None):
        if min_area is not None:
            self.min_area = min_area
        if max_area is not None:
            self.max_area = max_area