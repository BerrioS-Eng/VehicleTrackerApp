"""
analyzer.py
Responsabilidad: análisis cinemático del vehículo rastreado.
Posición, velocidad y aceleración mediante diferencias finitas.
Suavizado de datos y conversión de escala px ↔ metros.
"""

import math


class Analyzer:
    """Análisis de movimiento sobre el centroide detectado."""

    def __init__(self, fps=30, smooth_window=5):
        self.fps = fps
        self.smooth_window = smooth_window
        self._positions = []

        # Escala: puntos A y B de referencia
        self._point_a = None  # (x, y) en px
        self._point_b = None  # (x, y) en px
        self._real_distance_m = None
        self._px_per_meter = None

    # ── Escala ─────────────────────────────────────────────

    def set_scale(self, point_a, point_b, real_distance_m):
        """
        Calibra la escala con dos puntos y su distancia real.
        point_a, point_b: (x, y) en píxeles.
        real_distance_m: distancia real en metros.
        """
        self._point_a = point_a
        self._point_b = point_b
        self._real_distance_m = real_distance_m

        dx = point_b[0] - point_a[0]
        dy = point_b[1] - point_a[1]
        px_dist = math.sqrt(dx ** 2 + dy ** 2)

        if real_distance_m > 0:
            self._px_per_meter = px_dist / real_distance_m

    def has_scale(self) -> bool:
        return self._px_per_meter is not None

    def get_scale_points(self):
        """Retorna (point_a, point_b) o (None, None)."""
        return self._point_a, self._point_b

    def get_real_distance(self):
        return self._real_distance_m

    def get_px_per_meter(self):
        return self._px_per_meter

    def px_to_meters(self, px_value):
        if self._px_per_meter is None or self._px_per_meter == 0:
            return None
        return px_value / self._px_per_meter

    def clear_scale(self):
        self._point_a = None
        self._point_b = None
        self._real_distance_m = None
        self._px_per_meter = None

    # ── Registro de posiciones ─────────────────────────────

    def add_position(self, frame_num, centroid):
        cx, cy = centroid
        dt = 1 / self.fps if self.fps > 0 else 0
        time_sec = frame_num * dt

        self._positions.append({
            "frame": frame_num,
            "time": time_sec,
            "cx": cx,
            "cy": cy,
        })

    def clear_positions(self):
        self._positions = []

    def get_position_count(self) -> int:
        return len(self._positions)

    def get_last_position(self):
        if not self._positions:
            return None
        return self._positions[-1]

    # ── Suavizado ──────────────────────────────────────────

    def _smooth(self, data):
        n = len(data)
        if n < self.smooth_window:
            return data[:]

        half = self.smooth_window // 2
        smoothed = []

        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            avg = sum(data[start:end]) / (end - start)
            smoothed.append(avg)

        return smoothed

    # ── Datos de posición ──────────────────────────────────

    def get_time_array(self):
        return [p["time"] for p in self._positions]

    def get_positions_px(self):
        cx = [p["cx"] for p in self._positions]
        cy = [p["cy"] for p in self._positions]
        return cx, cy

    def get_positions_smooth(self):
        cx, cy = self.get_positions_px()
        return self._smooth(cx), self._smooth(cy)

    def get_trajectory(self):
        """Retorna lista de (cx, cy) para dibujar la trayectoria."""
        return [(p["cx"], p["cy"]) for p in self._positions]

    # ── Velocidad ──────────────────────────────────────────

    def get_velocity(self):
        n = len(self._positions)
        if n < 2:
            return [], [], [], []

        cx, cy = self.get_positions_smooth()
        t_arr = self.get_time_array()

        t, vx, vy, vmag = [], [], [], []

        for i in range(n):
            if i < n - 1:
                dt = t_arr[i + 1] - t_arr[i]
                dx = cx[i + 1] - cx[i]
                dy = cy[i + 1] - cy[i]
            else:
                dt = t_arr[i] - t_arr[i - 1]
                dx = cx[i] - cx[i - 1]
                dy = cy[i] - cy[i - 1]

            if dt == 0:
                vx_i, vy_i, vm_i = 0, 0, 0
            else:
                vx_i = dx / dt
                vy_i = dy / dt
                vm_i = math.sqrt(vx_i ** 2 + vy_i ** 2)

            t.append(t_arr[i])
            vx.append(vx_i)
            vy.append(vy_i)
            vmag.append(vm_i)

        return t, self._smooth(vx), self._smooth(vy), self._smooth(vmag)

    def get_current_velocity(self):
        """Retorna la velocidad instantánea actual (último frame)."""
        _, vx, vy, vmag = self.get_velocity()
        if not vmag:
            return 0, 0, 0
        return vx[-1], vy[-1], vmag[-1]

    # ── Aceleración ────────────────────────────────────────

    def get_acceleration(self):
        t_v, vx, vy, _ = self.get_velocity()
        n = len(vx)
        if n < 2:
            return [], [], [], []

        t, ax, ay, amag = [], [], [], []

        for i in range(n):
            if i < n - 1:
                dt = t_v[i + 1] - t_v[i]
                dvx = vx[i + 1] - vx[i]
                dvy = vy[i + 1] - vy[i]
            else:
                dt = t_v[i] - t_v[i - 1]
                dvx = vx[i] - vx[i - 1]
                dvy = vy[i] - vy[i - 1]

            if dt == 0:
                ax_i, ay_i, am_i = 0, 0, 0
            else:
                ax_i = dvx / dt
                ay_i = dvy / dt
                am_i = math.sqrt(ax_i ** 2 + ay_i ** 2)

            t.append(t_v[i])
            ax.append(ax_i)
            ay.append(ay_i)
            amag.append(am_i)

        return t, self._smooth(ax), self._smooth(ay), self._smooth(amag)

    def get_current_acceleration(self):
        """Retorna la aceleración instantánea actual."""
        _, ax, ay, amag = self.get_acceleration()
        if not amag:
            return 0, 0, 0
        return ax[-1], ay[-1], amag[-1]