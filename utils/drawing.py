"""
drawing.py
Responsabilidad: funciones auxiliares para dibujar
contornos, centroides, trayectoria, velocidad,
marcadores de escala y etiquetas sobre frames.
"""

import cv2
import math

# ── Colores BGR ────────────────────────────────────────────
COLOR_CONTOUR = (0, 255, 0)
COLOR_CENTROID = (255, 100, 0)
COLOR_TRAJECTORY = (0, 255, 255)
COLOR_VELOCITY = (0, 200, 255)
COLOR_SCALE_A = (0, 255, 0)
COLOR_SCALE_B = (0, 0, 255)
COLOR_SCALE_LINE = (255, 255, 0)
COLOR_TEXT = (255, 255, 255)


def draw_contours(frame, contours, color=COLOR_CONTOUR, thickness=2):
    cv2.drawContours(frame, contours, -1, color, thickness)
    return frame


def draw_centroids(frame, centroids, color=COLOR_CENTROID, radius=8):
    for (cx, cy) in centroids:
        cv2.circle(frame, (cx, cy), radius, color, -1)
        cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 2)
    return frame


def draw_labels(frame, centroids, prefix="V", color=COLOR_TEXT):
    for i, (cx, cy) in enumerate(centroids):
        label = f"{prefix}{i + 1}"
        cv2.putText(
            frame, label, (cx + 15, cy - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )
    return frame


def draw_trajectory(frame, trajectory, color=COLOR_TRAJECTORY, thickness=2):
    if len(trajectory) < 2:
        return frame
    for i in range(1, len(trajectory)):
        pt1 = trajectory[i - 1]
        pt2 = trajectory[i]
        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
    return frame


def draw_velocity_info(frame, velocity_mag, position, has_scale=False,
                       px_per_meter=None):
    if position is None:
        return frame

    cx, cy = position

    text_px = f"{velocity_mag:.1f} px/s"
    cv2.putText(
        frame, text_px, (cx + 20, cy + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_VELOCITY, 3,
    )

    if has_scale and px_per_meter and px_per_meter > 0:
        v_ms = velocity_mag / px_per_meter
        text_m = f"{v_ms:.2f} m/s"
        cv2.putText(
            frame, text_m, (cx + 20, cy + 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 3,
        )

    return frame


def draw_scale_markers(frame, point_a, point_b, real_distance_m):
    if point_a is None or point_b is None:
        return frame

    ax, ay = point_a
    bx, by = point_b

    # Línea A↔B
    cv2.line(frame, (ax, ay), (bx, by), COLOR_SCALE_LINE, 3, cv2.LINE_AA)

    # Punto A
    cv2.circle(frame, (ax, ay), 10, COLOR_SCALE_A, -1)
    cv2.circle(frame, (ax, ay), 10, (255, 255, 255), 2)
    cv2.putText(
        frame, "A", (ax + 15, ay - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_SCALE_A, 3,
    )

    # Punto B
    cv2.circle(frame, (bx, by), 10, COLOR_SCALE_B, -1)
    cv2.circle(frame, (bx, by), 10, (255, 255, 255), 2)
    cv2.putText(
        frame, "B", (bx + 15, by - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_SCALE_B, 3,
    )

    # Distancia en el punto medio
    mx = (ax + bx) // 2
    my = (ay + by) // 2
    px_dist = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
    label = f"{real_distance_m:.1f}m ({px_dist:.0f}px)"
    cv2.putText(
        frame, label, (mx + 15, my - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_SCALE_LINE, 3,
    )

    return frame


def draw_detections(frame, detections):
    draw_contours(frame, detections["contours"])
    draw_centroids(frame, detections["centroids"])
    draw_labels(frame, detections["centroids"])
    return frame