# Vehicle Tracker

Aplicación de escritorio para detección y análisis cinemático de vehículos en video, usando técnicas de procesamiento digital de imágenes.

## Características

- Reproducción de video con controles play/pausa
- Sustracción de fondo (MOG2 / KNN) con operaciones morfológicas configurables
- Detección de vehículos por análisis de contornos (área, aspect ratio, solidez)
- ROI interactiva (polígono dibujado sobre el video)
- Filtrado por color HSV
- Calibración de escala px ↔ metros (puntos A↔B)
- Trayectoria y velocidad en tiempo real sobre el video
- Gráficas de posición, velocidad y aceleración (matplotlib)
- Interfaz Tkinter con layout 2x2 y tema oscuro

## Estructura

```
vehicle_tracker/
├── main.py                  # Punto de entrada
├── config/settings.py       # Constantes globales
├── core/
│   ├── video_handler.py     # Lectura y reproducción de video
│   ├── preprocessor.py      # Color, BG subtraction, morfología, ROI
│   ├── detector.py          # Detección por contornos
│   └── analyzer.py          # Análisis cinemático (pos, vel, acel)
├── ui/main_window.py        # Ventana principal (Tkinter)
├── utils/drawing.py         # Dibujo de anotaciones sobre frames
└── resources/               # Videos de prueba (.gitignore)
```

## Requisitos

- Python 3.10+
- OpenCV, NumPy, Pillow, Matplotlib

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
python main.py
```
