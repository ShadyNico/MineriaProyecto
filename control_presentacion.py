"""Paso 3 - Control de PowerPoint con gestos en tiempo real.

Cámara -> MediaPipe localiza la mano -> recorte -> CNN clasifica -> tecla.

Para que UN gesto no cambie VARIAS diapositivas se usan tres filtros:
  1. Umbral de confianza: la CNN debe estar segura (def. 85 %).
  2. Estabilidad: el mismo gesto en N frames seguidos (def. 6).
  3. Bloqueo + espera: tras disparar, ese gesto queda bloqueado hasta que
     desaparezca (bajas la mano o haces otro gesto), y además hay un
     tiempo mínimo entre acciones (def. 1.2 s).

Uso:
    python control_presentacion.py --simular   # prueba: no pulsa teclas
    python control_presentacion.py             # controla PowerPoint
Después de lanzarlo, haz clic en la ventana de PowerPoint para que reciba
las teclas. Q o ESC en la ventana de la cámara para salir.
"""
import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # oculta mensajes informativos de TF
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
from collections import deque

import cv2
import numpy as np
import pyautogui
from tensorflow import keras

from config import CAMARA, GESTOS, RUTA_CLASES, RUTA_MODELO, TAM_IMAGEN
from detector_mano import (
    DetectorMano,
    abrir_camara,
    caja_cuadrada,
    dibujar_mano,
    pegar_miniatura,
    preprocesar,
    recortar_mano,
)

VENTANA = "GestureControl CNN"
NOMBRE_TECLA = {"right": "->", "left": "<-", "f5": "F5", "esc": "ESC"}


def texto(img, msg, pos, color=(255, 255, 255), escala=0.6, grosor=1):
    cv2.putText(img, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grosor + 3)
    cv2.putText(img, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor)


def barras_probabilidad(img, clases, probs, umbral, x=10, y=100):
    for i, c in enumerate(clases):
        p = 0.0 if probs is None else float(probs[i])
        yy = y + i * 24
        color = (0, 220, 0) if p >= umbral else (0, 165, 255)
        cv2.rectangle(img, (x + 95, yy - 13), (x + 95 + int(150 * p), yy + 3), color, -1)
        cv2.rectangle(img, (x + 95, yy - 13), (x + 245, yy + 3), (255, 255, 255), 1)
        texto(img, c, (x, yy), escala=0.5)
        texto(img, f"{p * 100:4.0f}%", (x + 252, yy), escala=0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--umbral", type=float, default=0.85, help="confianza mínima de la CNN (def. 0.85)")
    ap.add_argument("--frames", type=int, default=6, help="frames seguidos con el mismo gesto (def. 6)")
    ap.add_argument("--espera", type=float, default=1.2, help="segundos mínimos entre acciones (def. 1.2)")
    ap.add_argument("--simular", action="store_true", help="no pulsa teclas, solo muestra la acción")
    ap.add_argument("--camara", type=int, default=CAMARA)
    args = ap.parse_args()

    if not RUTA_MODELO.exists():
        raise SystemExit("No hay modelo entrenado. Ejecuta primero entrenar_cnn.py")
    clases = json.loads(RUTA_CLASES.read_text(encoding="utf-8"))
    modelo = keras.models.load_model(RUTA_MODELO, compile=False)
    modelo(np.zeros((1, TAM_IMAGEN, TAM_IMAGEN, 1), np.float32), training=False)  # calentamiento
    pyautogui.PAUSE = 0

    print("Clases:", ", ".join(f"{c} -> {GESTOS.get(c) or 'sin accion'}" for c in clases))
    print("SIMULACION: no se pulsarán teclas." if args.simular
          else "Haz clic en PowerPoint para que reciba las teclas.")

    historial = deque(maxlen=args.frames)
    bloqueado = None
    ultimo_disparo = 0.0
    aviso, aviso_hasta = "", 0.0
    t_prev = time.monotonic()
    fps = 0.0

    cap = abrir_camara(args.camara)
    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENTANA, 640, 480)
    cv2.setWindowProperty(VENTANA, cv2.WND_PROP_TOPMOST, 1)  # visible sobre PowerPoint

    with DetectorMano() as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer la cámara.")
                break
            frame = cv2.flip(frame, 1)
            vista = frame.copy()

            # --- Detección + clasificación ---
            probs, gesto = None, None
            puntos = detector.detectar(frame)
            if puntos is not None:
                caja = caja_cuadrada(puntos)
                img = preprocesar(recortar_mano(frame, caja))
                x = img[np.newaxis, ..., np.newaxis].astype(np.float32)
                probs = modelo(x, training=False).numpy()[0]
                i = int(probs.argmax())
                if probs[i] >= args.umbral:
                    gesto = clases[i]
                dibujar_mano(vista, puntos, caja, (0, 255, 0) if gesto else (0, 165, 255))
                pegar_miniatura(vista, img)

            # --- Anti-repetición ---
            historial.append(gesto)
            if bloqueado is not None and bloqueado not in historial:
                bloqueado = None  # el gesto desapareció: se puede volver a usar
            estable = (
                historial[0]
                if len(historial) == historial.maxlen
                and historial[0] is not None
                and historial.count(historial[0]) == len(historial)
                else None
            )

            ahora = time.monotonic()
            tecla = GESTOS.get(estable) if estable else None
            if tecla and estable != bloqueado and ahora - ultimo_disparo >= args.espera:
                try:
                    if not args.simular:
                        pyautogui.press(tecla)
                    aviso = f"{estable.upper()}  [{NOMBRE_TECLA.get(tecla, tecla)}]"
                    print(time.strftime("%H:%M:%S"), aviso)
                except pyautogui.FailSafeException:
                    aviso = "FailSafe: mouse en una esquina"
                aviso_hasta = ahora + 1.0
                ultimo_disparo = ahora
                bloqueado = estable

            # --- Interfaz ---
            fps = 0.9 * fps + 0.1 / max(ahora - t_prev, 1e-6)
            t_prev = ahora
            if puntos is None:
                estado, color = "Sin mano", (200, 200, 200)
            elif bloqueado:
                estado, color = f"Bloqueado: {bloqueado} (cambia o baja la mano)", (0, 165, 255)
            elif ahora - ultimo_disparo < args.espera:
                estado, color = f"Espera {args.espera - (ahora - ultimo_disparo):.1f}s", (0, 165, 255)
            else:
                n = historial.count(gesto) if gesto else 0
                estado, color = f"Gesto: {gesto or '?'}  {n}/{args.frames}", (0, 255, 0)

            texto(vista, ("SIMULACION  " if args.simular else "") + f"FPS {fps:.0f}", (10, 28), (0, 255, 255), 0.6, 2)
            texto(vista, estado, (10, 58), color, 0.6, 2)
            barras_probabilidad(vista, clases, probs, args.umbral)
            if ahora < aviso_hasta:
                h, w = vista.shape[:2]
                texto(vista, aviso, (10, h - 20), (0, 255, 0), 1.0, 2)

            cv2.imshow(VENTANA, vista)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
