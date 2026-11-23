"""Localización y recorte de la mano con MediaPipe.

MediaPipe SOLO encuentra dónde está la mano (21 puntos clave). Con esos
puntos recortamos un cuadrado alrededor de la mano y ese recorte es lo que
clasifica la CNN. Es decir: MediaPipe localiza, la CNN reconoce el gesto.

Funciona con las dos APIs de MediaPipe:
  * la antigua `mp.solutions.hands` (mediapipe <= 0.10.2x)
  * la nueva `mediapipe.tasks` HandLandmarker (descarga el modelo una vez)
"""
from __future__ import annotations

import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

from config import (
    ALTO_CAMARA,
    ANCHO_CAMARA,
    MARGEN_RECORTE,
    RUTA_LANDMARKER,
    TAM_IMAGEN,
)

URL_LANDMARKER = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Conexiones entre los 21 puntos de la mano (solo para dibujar).
CONEXIONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class DetectorMano:
    """Devuelve los 21 puntos (en píxeles) de la primera mano detectada."""

    def __init__(self, confianza: float = 0.6):
        self._ultimo_ts = 0
        soluciones = getattr(mp, "solutions", None)
        if soluciones is not None and hasattr(soluciones, "hands"):
            self._modo = "solutions"
            self._hands = soluciones.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=confianza,
                min_tracking_confidence=0.5,
            )
        else:
            self._modo = "tasks"
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision

            if not RUTA_LANDMARKER.exists():
                RUTA_LANDMARKER.parent.mkdir(parents=True, exist_ok=True)
                print("Descargando modelo de MediaPipe (solo la primera vez)...")
                urllib.request.urlretrieve(URL_LANDMARKER, RUTA_LANDMARKER)
            # Se pasa el modelo como bytes: evita problemas con rutas con
            # acentos (p. ej. "Minería") en Windows.
            opciones = vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(
                    model_asset_buffer=RUTA_LANDMARKER.read_bytes()
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=confianza,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(opciones)

    def detectar(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._modo == "solutions":
            rgb.flags.writeable = False
            res = self._hands.process(rgb)
            if not res.multi_hand_landmarks:
                return None
            puntos = res.multi_hand_landmarks[0].landmark
        else:
            # El modo VIDEO exige marcas de tiempo estrictamente crecientes.
            ts = max(int(time.monotonic() * 1000), self._ultimo_ts + 1)
            self._ultimo_ts = ts
            imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self._landmarker.detect_for_video(imagen, ts)
            if not res.hand_landmarks:
                return None
            puntos = res.hand_landmarks[0]
        return np.array([[p.x * w, p.y * h] for p in puntos], dtype=np.float32)

    def cerrar(self) -> None:
        if self._modo == "solutions":
            self._hands.close()
        else:
            self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cerrar()


def abrir_camara(indice: int) -> cv2.VideoCapture:
    # CAP_DSHOW abre la cámara mucho más rápido en Windows.
    cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(indice)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {indice}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO_CAMARA)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO_CAMARA)
    return cap


def caja_cuadrada(puntos: np.ndarray, margen: float = MARGEN_RECORTE):
    """Cuadrado (x1, y1, x2, y2) que contiene la mano más un margen."""
    x_min, y_min = puntos.min(axis=0)
    x_max, y_max = puntos.max(axis=0)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    lado = max(x_max - x_min, y_max - y_min) * (1 + 2 * margen)
    lado = max(lado, 40)  # evita recortes diminutos
    mitad = lado / 2
    return (int(cx - mitad), int(cy - mitad), int(cx + mitad), int(cy + mitad))


def recortar_mano(frame: np.ndarray, caja) -> np.ndarray:
    """Recorta la caja; si se sale de la imagen, rellena replicando bordes."""
    x1, y1, x2, y2 = caja
    h, w = frame.shape[:2]
    izq, arr = max(0, -x1), max(0, -y1)
    der, aba = max(0, x2 - w), max(0, y2 - h)
    if izq or arr or der or aba:
        frame = cv2.copyMakeBorder(frame, arr, aba, izq, der, cv2.BORDER_REPLICATE)
        x1, x2, y1, y2 = x1 + izq, x2 + izq, y1 + arr, y2 + arr
    return frame[y1:y2, x1:x2]


def preprocesar(recorte_bgr: np.ndarray) -> np.ndarray:
    """Recorte BGR -> imagen gris TAM_IMAGEN x TAM_IMAGEN (uint8).

    Se usa idéntica al capturar el dataset y al predecir en vivo.
    """
    gris = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gris, (TAM_IMAGEN, TAM_IMAGEN), interpolation=cv2.INTER_AREA)


def dibujar_mano(img, puntos, caja, color=(0, 255, 0)) -> None:
    for a, b in CONEXIONES:
        pa = tuple(int(v) for v in puntos[a])
        pb = tuple(int(v) for v in puntos[b])
        cv2.line(img, pa, pb, (255, 255, 255), 2)
    for p in puntos:
        cv2.circle(img, (int(p[0]), int(p[1])), 3, color, -1)
    cv2.rectangle(img, caja[:2], caja[2:], color, 2)


def pegar_miniatura(img, gris: np.ndarray, tam: int = 128) -> None:
    """Muestra en la esquina superior derecha lo que ve la CNN."""
    mini = cv2.cvtColor(cv2.resize(gris, (tam, tam)), cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    img[8:8 + tam, w - tam - 8:w - 8] = mini
    cv2.rectangle(img, (w - tam - 9, 7), (w - 7, 8 + tam), (255, 255, 255), 1)


def guardar_png(ruta, imagen) -> None:
    """cv2.imwrite falla en silencio con rutas con acentos en Windows."""
    ok, buffer = cv2.imencode(".png", imagen)
    if not ok:
        raise RuntimeError(f"No se pudo codificar {ruta}")
    buffer.tofile(str(ruta))


def leer_gris(ruta) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"No se pudo leer {ruta}")
    if img.shape != (TAM_IMAGEN, TAM_IMAGEN):
        img = cv2.resize(img, (TAM_IMAGEN, TAM_IMAGEN), interpolation=cv2.INTER_AREA)
    return img
