# GestureControl CNN

Sistema para controlar presentaciones de PowerPoint con gestos de la mano usando
una **Red Neuronal Convolucional (CNN)**.

```
Cámara ─► OpenCV + MediaPipe ─► recorte de la mano ─► CNN (Keras) ─► PyAutoGUI ─► PowerPoint
          (localiza la mano)     (128x128, gris)      (clasifica)     (pulsa tecla)
```

MediaPipe **solo localiza** la mano; quien **reconoce el gesto es la CNN** entrenada
con nuestro propio dataset.

| Gesto | Clase | Acción en PowerPoint |
|---|---|---|
| Mano abierta | `siguiente` | → (siguiente diapositiva) |
| Puño cerrado | `anterior` | ← (diapositiva anterior) |
| Dos dedos (V) | `iniciar` | F5 (iniciar presentación) |
| Pulgar arriba | `finalizar` | Esc (finalizar presentación) |
| Mano relajada / otra pose | `nada` | sin acción |

La clase `nada` es importante: le enseña a la CNN qué **no** es un gesto, para
que no cambie diapositivas mientras gesticulas al hablar.

## Archivos

| Archivo | Qué hace |
|---|---|
| `config.py` | Gestos, teclas, tamaño de imagen y rutas (todo se configura aquí) |
| `detector_mano.py` | MediaPipe + recorte y preprocesamiento de la mano |
| `capturar_dataset.py` | **Paso 1**: captura el dataset con la cámara |
| `entrenar_cnn.py` | **Paso 2**: entrena la CNN y genera métricas y gráficas |
| `control_presentacion.py` | **Paso 3**: controla PowerPoint en tiempo real |

## Instalación

TensorFlow y MediaPipe necesitan **Python 3.10 – 3.12** (no funcionan con 3.15).
Con `uv` (ya instalado en este equipo):

```powershell
uv venv --python 3.12 .venv
```

```powershell
.venv\Scripts\activate
```

```powershell
uv pip install -r requirements.txt
```

## Uso

### 1. Capturar el dataset

```powershell
python capturar_dataset.py
```

- Teclas `1`…`5` eligen el gesto, `ESPACIO` graba un lote de 200 imágenes, `Q` sale.
- En la esquina superior derecha ves **exactamente lo que verá la CNN**.
- Graba **2–3 lotes por gesto** (~400–600 imágenes) moviendo la mano: más cerca/lejos,
  algo girada, a la izquierda/derecha, con otra luz. Para `nada` haz poses variadas
  (mano de lado, dedos a medio cerrar, señalar, etc.).
- Captura el **último lote de cada gesto en condiciones distintas**: ese 20 % final se
  usa como validación.

### 2. Entrenar la CNN

```powershell
python entrenar_cnn.py
```

Genera en `modelo/`: `gestos_cnn.keras`, `clases.json`, `curvas_entrenamiento.png`,
`matriz_confusion.png` y `reporte.txt` (precision / recall / F1 por clase), útiles para
el informe.

Arquitectura: 3 bloques `Conv-BN-ReLU ×2 + MaxPool + Dropout` (32/64/128 filtros),
una `Conv 256`, `GlobalAveragePooling`, `Dense 128` y `softmax`. Incluye aumentación de
datos (volteo, rotación, zoom, traslación, brillo y contraste), *early stopping* y
pesos por clase.

### 3. Controlar la presentación

Primero pruébalo sin pulsar teclas:

```powershell
python control_presentacion.py --simular
```

Después, con PowerPoint abierto:

```powershell
python control_presentacion.py
```

**Haz clic en la ventana de PowerPoint** después de lanzar el script: las teclas van a
la ventana que tenga el foco. La ventana de la cámara queda siempre visible encima.

#### Cómo se evita que un gesto pase varias diapositivas

1. **Umbral de confianza** (`--umbral 0.85`): la CNN debe estar segura.
2. **Estabilidad** (`--frames 6`): el mismo gesto en 6 frames seguidos.
3. **Bloqueo**: tras disparar, ese gesto no vuelve a actuar hasta que bajas la mano o
   haces otro gesto.
4. **Tiempo de espera** (`--espera 1.2`): mínimo 1.2 s entre acciones.

Si reacciona lento, baja `--frames`; si se dispara solo, sube `--umbral`.

## Recomendación para la demo

Empieza con 3 clases (`siguiente`, `anterior`, `nada`): en `config.py` comenta
`iniciar` y `finalizar`, captura, entrena y prueba. Cuando funcione bien, descoméntalas,
captura esos dos gestos y vuelve a entrenar.

## Problemas comunes

- **No detecta la mano**: más luz y fondo sin mucho desorden; la mano completa en cuadro.
- **Confunde dos gestos**: mira `matriz_confusion.png` y captura más imágenes variadas
  de esos dos gestos.
- **Funciona en validación pero no en vivo**: captura con la misma cámara, distancia y
  luz que tendrás en la presentación.
- **Las teclas no llegan a PowerPoint**: haz clic en PowerPoint (no en la ventana de la
  cámara). Ojo: si la ventana de la cámara tiene el foco, el gesto `finalizar` (Esc)
  cierra el programa.
