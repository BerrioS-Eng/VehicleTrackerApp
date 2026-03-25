"""
drawing.py
Responsabilidad: funciones auxiliares para dibujar
contornos, centroides y etiquetas sobre frames.
"""

import cv2

# ── Colores BGR ────────────────────────────────────────────
COLOR_CONTOUR = (0, 255, 0)      # Verde
COLOR_CENTROID = (255, 100, 0)   # Azul
COLOR_TEXT = (255, 255, 255)     # Blanco


def draw_contours(frame, contours, color=COLOR_CONTOUR, thickness=2):
    """Dibuja los contornos sobre el frame."""
    cv2.drawContours(frame, contours, -1, color, thickness)
    return frame


def draw_centroids(frame, centroids, color=COLOR_CENTROID, radius=6):
    """Dibuja los centroides como círculos sobre el frame."""
    for (cx, cy) in centroids:
        cv2.circle(frame, (cx, cy), radius, color, -1)
        # Borde blanco para mejor visibilidad
        cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 1)
    return frame


def draw_labels(frame, centroids, prefix="V", color=COLOR_TEXT):
    """Dibuja etiquetas con ID junto a cada centroide."""
    for i, (cx, cy) in enumerate(centroids):
        label = f"{prefix}{i + 1}"
        cv2.putText(
            frame, label, (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )
    return frame


def draw_detections(frame, detections):
    """
    Dibuja contornos + centroides + etiquetas sobre el frame.
    detections: dict retornado por Detector.detect().
    """
    draw_contours(frame, detections["contours"])
    draw_centroids(frame, detections["centroids"])
    draw_labels(frame, detections["centroids"])
    return frame