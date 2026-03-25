"""
drawing.py
Responsabilidad: funciones auxiliares para dibujar
contornos, bounding boxes, centroides y etiquetas sobre frames.
"""

import cv2


# ── Colores BGR por defecto ────────────────────────────────
COLOR_CONTOUR = (0, 255, 0)      # Verde
COLOR_BBOX = (255, 0, 0)         # Azul
COLOR_CENTROID = (0, 0, 255)     # Rojo
COLOR_TEXT = (255, 255, 255)     # Blanco


def draw_contours(frame, contours, color=COLOR_CONTOUR, thickness=2):
    """Dibuja los contornos sobre el frame."""
    cv2.drawContours(frame, contours, -1, color, thickness)
    return frame


def draw_bounding_boxes(frame, bboxes, color=COLOR_BBOX, thickness=2):
    """Dibuja los bounding boxes (x, y, w, h) sobre el frame."""
    for (x, y, w, h) in bboxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    return frame


def draw_centroids(frame, centroids, color=COLOR_CENTROID, radius=5):
    """Dibuja los centroides como círculos sobre el frame."""
    for (cx, cy) in centroids:
        cv2.circle(frame, (cx, cy), radius, color, -1)
    return frame


def draw_labels(frame, bboxes, prefix="V", color=COLOR_TEXT):
    """Dibuja etiquetas con ID sobre cada bounding box."""
    for i, (x, y, w, h) in enumerate(bboxes):
        label = f"{prefix}{i + 1}"
        cv2.putText(
            frame, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
    return frame


def draw_detections(frame, detections):
    """
    Dibuja toda la información de detección sobre el frame.
    detections: dict retornado por Detector.detect().
    Retorna el frame anotado (modifica el original).
    """
    draw_contours(frame, detections["contours"])
    draw_bounding_boxes(frame, detections["bounding_boxes"])
    draw_centroids(frame, detections["centroids"])
    draw_labels(frame, detections["bounding_boxes"])
    return frame