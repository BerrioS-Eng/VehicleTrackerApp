"""
analyzer.py
Responsabilidad: análisis cinemático del vehículo rastreado.
Posición, velocidad y aceleración mediante diferencias finitas.
Suavizado de datos para reducir ruido en las derivadas.
"""

import math


class Analyzer:
    """Análisis de movimiento sobre el centroide detectado."""

    def __init__(self, fps=30, smooth_window=5):
        self.fps = fps
        self.smooth_window = smooth_window  # Ventana de media móvil
        self._positions = []

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
        """
        Media móvil centrada para suavizar datos.
        Reduce el ruido de detección antes de calcular derivadas.
        """
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
        """Retorna posiciones crudas (sin suavizar)."""
        cx = [p["cx"] for p in self._positions]
        cy = [p["cy"] for p in self._positions]
        return cx, cy

    def get_positions_smooth(self):
        """Retorna posiciones suavizadas."""
        cx, cy = self.get_positions_px()
        return self._smooth(cx), self._smooth(cy)

    # ── Velocidad (diferencias finitas hacia adelante) ─────

    def get_velocity(self):
        """
        Velocidad instantánea sobre posiciones suavizadas:
          vx[i] = (x[i+1] - x[i]) / dt

        Retorna (t, vx, vy, vmag).
        """
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

        # Suavizar también la velocidad resultante
        return t, self._smooth(vx), self._smooth(vy), self._smooth(vmag)

    # ── Aceleración (segunda derivada numérica) ────────────

    def get_acceleration(self):
        """
        Aceleración como derivada de la velocidad suavizada:
          ax[i] = (vx[i+1] - vx[i]) / dt

        Retorna (t, ax, ay, amag).
        """
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

        # Suavizar la aceleración también
        return t, self._smooth(ax), self._smooth(ay), self._smooth(amag)