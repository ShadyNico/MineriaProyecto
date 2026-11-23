"""Paso 1 - Captura del dataset propio con la cámara.

Controles en la ventana:
  1..N     elegir el gesto a capturar
  ESPACIO  empezar / detener la grabación de un lote
  Q o ESC  salir

Cada lote guarda hasta --lote imágenes (recortes de la mano en gris,
128x128) en dataset/<gesto>/. Graba 2 o 3 lotes por gesto cambiando la
posición de la mano, la distancia, la inclinación y la iluminación.

Uso:
    python capturar_dataset.py
    python capturar_dataset.py --lote 150 --intervalo 3
"""
import argparse
import time

import cv2

from config import CAMARA, DESCRIPCION, DIR_DATASET, GESTOS
from detector_mano import (
    DetectorMano,
    abrir_camara,
    caja_cuadrada,
    dibujar_mano,
    guardar_png,
    pegar_miniatura,
    preprocesar,
    recortar_mano,
)

VENTANA = "GestureControl CNN - Captura de dataset"


def texto(img, msg, pos, color=(255, 255, 255), escala=0.6, grosor=1):
    cv2.putText(img, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grosor + 3)
    cv2.putText(img, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lote", type=int, default=200, help="imágenes por lote (def. 200)")
    ap.add_argument("--intervalo", type=int, default=2,
                    help="guardar 1 de cada N frames para no tener fotos casi iguales (def. 2)")
    ap.add_argument("--camara", type=int, default=CAMARA)
    args = ap.parse_args()

    nombres = list(GESTOS)
    for g in nombres:
        (DIR_DATASET / g).mkdir(parents=True, exist_ok=True)
    conteos = {g: len(list((DIR_DATASET / g).glob("*.png"))) for g in nombres}

    actual, grabando, en_lote, n_frame = 0, False, 0, 0
    cap = abrir_camara(args.camara)

    with DetectorMano() as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer la cámara.")
                break
            frame = cv2.flip(frame, 1)  # efecto espejo (más natural)
            vista = frame.copy()        # se dibuja sobre la copia, no sobre lo que se guarda
            gesto = nombres[actual]

            puntos = detector.detectar(frame)
            if puntos is not None:
                caja = caja_cuadrada(puntos)
                img = preprocesar(recortar_mano(frame, caja))
                dibujar_mano(vista, puntos, caja, (0, 0, 255) if grabando else (0, 255, 0))
                pegar_miniatura(vista, img)

                if grabando and n_frame % args.intervalo == 0:
                    nombre = f"{gesto}_{time.strftime('%Y%m%d_%H%M%S')}_{conteos[gesto]:05d}.png"
                    guardar_png(DIR_DATASET / gesto / nombre, img)
                    conteos[gesto] += 1
                    en_lote += 1
                    if en_lote >= args.lote:
                        grabando = False
                        print(f"Lote terminado: {gesto} = {conteos[gesto]} imágenes")
            n_frame += 1

            # --- Interfaz ---
            estado = f"GRABANDO {en_lote}/{args.lote}" if grabando else "PAUSA (ESPACIO = grabar)"
            texto(vista, f"Gesto: {gesto} - {DESCRIPCION.get(gesto, '')}", (10, 28), (0, 255, 255), 0.7, 2)
            texto(vista, estado, (10, 56), (0, 0, 255) if grabando else (200, 200, 200), 0.6, 2)
            if puntos is None:
                texto(vista, "Sin mano detectada", (10, 84), (0, 165, 255))
            y = vista.shape[0] - 12 - 22 * (len(nombres) - 1)
            for i, g in enumerate(nombres):
                color = (0, 255, 255) if i == actual else (255, 255, 255)
                texto(vista, f"[{i + 1}] {g}: {conteos[g]}", (10, y + 22 * i), color, 0.55)

            cv2.imshow(VENTANA, vista)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == 32:  # espacio
                grabando = not grabando
                en_lote = 0
            elif ord("1") <= tecla < ord("1") + len(nombres):
                actual = tecla - ord("1")
                grabando = False

    cap.release()
    cv2.destroyAllWindows()
    print("\nResumen del dataset:")
    for g in nombres:
        print(f"  {g:<10} {conteos[g]:>5} imágenes")


if __name__ == "__main__":
    main()
