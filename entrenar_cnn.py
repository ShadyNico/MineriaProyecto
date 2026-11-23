"""Paso 2 - Entrenamiento de la CNN que clasifica los gestos.

Lee dataset/<gesto>/*.png, entrena una red convolucional y guarda en modelo/:
  gestos_cnn.keras          el modelo entrenado
  clases.json               el orden de las clases
  curvas_entrenamiento.png  accuracy y loss por época
  matriz_confusion.png      errores por clase en validación
  reporte.txt               precision / recall / F1 por clase

Validación: por cada gesto se reserva el ÚLTIMO 20 % de las imágenes
capturadas (orden cronológico). Así la validación no está llena de frames
casi idénticos a los de entrenamiento y el accuracy es más honesto.

Uso:
    python entrenar_cnn.py
    python entrenar_cnn.py --epocas 60 --lote 32
"""
import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # oculta mensajes informativos de TF
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import DIR_DATASET, DIR_MODELO, GESTOS, RUTA_CLASES, RUTA_MODELO, TAM_IMAGEN
from detector_mano import leer_gris

MIN_IMAGENES = 30

# Con pocos datos hay pocos pasos por época; con el momentum por defecto (0.99)
# las estadísticas de BatchNorm que se usan al validar/predecir tardan muchas
# épocas en estabilizarse y la red predice una sola clase en validación.
MOMENTUM_BN = 0.9


def bn():
    return layers.BatchNormalization(momentum=MOMENTUM_BN)


def cargar_dataset(clases, frac_val):
    x_tr, y_tr, x_va, y_va = [], [], [], []
    for idx, gesto in enumerate(clases):
        archivos = sorted((DIR_DATASET / gesto).glob("*.png"))
        n_val = max(1, int(round(len(archivos) * frac_val)))
        imagenes = [leer_gris(a) for a in archivos]
        x_tr += imagenes[:-n_val]
        y_tr += [idx] * (len(archivos) - n_val)
        x_va += imagenes[-n_val:]
        y_va += [idx] * n_val
        print(f"  {gesto:<10} total={len(archivos):>5}  entrenamiento={len(archivos) - n_val:>5}  validacion={n_val:>4}")
    a = lambda lista: np.stack(lista)[..., np.newaxis]  # (N, T, T, 1)
    return a(x_tr), np.array(y_tr), a(x_va), np.array(y_va)


def construir_cnn(n_clases):
    aumentacion = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),          # mano izquierda o derecha
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.10, 0.10),
            layers.RandomBrightness(0.25, value_range=(0, 255)),
            layers.RandomContrast(0.30),
        ],
        name="aumentacion",
    )

    entrada = keras.Input(shape=(TAM_IMAGEN, TAM_IMAGEN, 1), name="imagen")
    x = aumentacion(entrada)  # solo actúa durante el entrenamiento
    x = layers.Rescaling(1.0 / 255)(x)

    for filtros in (32, 64, 128):
        for _ in range(2):
            x = layers.Conv2D(filtros, 3, padding="same", use_bias=False)(x)
            x = bn()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.15)(x)

    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = bn()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    salida = layers.Dense(n_clases, activation="softmax", name="gesto")(x)

    modelo = keras.Model(entrada, salida, name="GestureControlCNN")
    modelo.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def graficar_curvas(hist, ruta):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(hist["accuracy"], label="entrenamiento")
    a1.plot(hist["val_accuracy"], label="validación")
    a1.set_title("Accuracy"); a1.set_xlabel("época"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(hist["loss"], label="entrenamiento")
    a2.plot(hist["val_loss"], label="validación")
    a2.set_title("Loss"); a2.set_xlabel("época"); a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ruta, dpi=120); plt.close(fig)


def graficar_confusion(cm, clases, ruta):
    fig, ax = plt.subplots(figsize=(1.3 * len(clases) + 2, 1.1 * len(clases) + 2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(clases)), clases, rotation=45, ha="right")
    ax.set_yticks(range(len(clases)), clases)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title("Matriz de confusión (validación)")
    limite = cm.max() / 2 if cm.max() else 0
    for i in range(len(clases)):
        for j in range(len(clases)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > limite else "black")
    fig.tight_layout(); fig.savefig(ruta, dpi=120); plt.close(fig)


def reporte(cm, clases):
    lineas = [f"{'clase':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'soporte':>10}"]
    for i, c in enumerate(clases):
        tp = cm[i, i]
        p = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        r = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        lineas.append(f"{c:<12}{p:>10.3f}{r:>10.3f}{f1:>10.3f}{int(cm[i, :].sum()):>10}")
    lineas.append(f"\naccuracy global: {np.trace(cm) / cm.sum():.3f}  ({int(np.trace(cm))}/{int(cm.sum())})")
    return "\n".join(lineas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epocas", type=int, default=50)
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--validacion", type=float, default=0.2, help="fracción para validación (def. 0.2)")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(42)

    clases = []
    for g in GESTOS:
        n = len(list((DIR_DATASET / g).glob("*.png")))
        if n >= MIN_IMAGENES:
            clases.append(g)
        else:
            print(f"AVISO: '{g}' tiene {n} imágenes (mínimo {MIN_IMAGENES}); se omite.")
    if len(clases) < 2:
        raise SystemExit("Se necesitan al menos 2 gestos con imágenes. Ejecuta primero capturar_dataset.py")

    print("Cargando dataset:")
    x_tr, y_tr, x_va, y_va = cargar_dataset(clases, args.validacion)

    ds_tr = (tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
             .shuffle(len(x_tr), seed=42).batch(args.lote).prefetch(tf.data.AUTOTUNE))
    ds_va = tf.data.Dataset.from_tensor_slices((x_va, y_va)).batch(args.lote).prefetch(tf.data.AUTOTUNE)

    # Pesos por clase por si alguna tiene menos imágenes.
    conteo = np.bincount(y_tr, minlength=len(clases))
    pesos = {i: len(y_tr) / (len(clases) * c) for i, c in enumerate(conteo)}

    DIR_MODELO.mkdir(parents=True, exist_ok=True)
    modelo = construir_cnn(len(clases))
    modelo.summary()

    # Las primeras épocas son inestables en validación: no se cortan ni se
    # baja el learning rate hasta que la red haya tenido tiempo de aprender.
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=12,
                                      start_from_epoch=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6,
                                          cooldown=2, min_lr=1e-5),
    ]
    hist = modelo.fit(ds_tr, validation_data=ds_va, epochs=args.epocas, shuffle=False,
                      class_weight=pesos, callbacks=callbacks, verbose=2)

    modelo.save(RUTA_MODELO)
    RUTA_CLASES.write_text(json.dumps(clases, ensure_ascii=False, indent=2), encoding="utf-8")

    pred = modelo.predict(x_va, verbose=0).argmax(axis=1)
    cm = np.zeros((len(clases), len(clases)), dtype=int)
    for real, p in zip(y_va, pred):
        cm[real, p] += 1

    texto = reporte(cm, clases)
    print("\n" + texto)
    (DIR_MODELO / "reporte.txt").write_text(texto, encoding="utf-8")
    graficar_curvas(hist.history, DIR_MODELO / "curvas_entrenamiento.png")
    graficar_confusion(cm, clases, DIR_MODELO / "matriz_confusion.png")
    print(f"\nModelo guardado en {RUTA_MODELO}")


if __name__ == "__main__":
    main()
