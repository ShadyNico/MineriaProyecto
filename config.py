"""Configuración central de GestureControl CNN.

Todos los scripts leen de aquí, así la captura, el entrenamiento y el
control en tiempo real usan exactamente las mismas clases y el mismo
preprocesamiento.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DIR_DATASET = RAIZ / "dataset"
DIR_MODELO = RAIZ / "modelo"
RUTA_MODELO = DIR_MODELO / "gestos_cnn.keras"
RUTA_CLASES = DIR_MODELO / "clases.json"
RUTA_LANDMARKER = DIR_MODELO / "hand_landmarker.task"

# ---------------------------------------------------------------------------
# Gestos = clases de la CNN, y la tecla que cada uno envía a PowerPoint.
# None significa "sin acción".
#
# Para empezar con 3 gestos (recomendado), comenta "iniciar" y "finalizar",
# captura solo siguiente / anterior / nada, entrena y prueba. Luego agrégalos.
# ---------------------------------------------------------------------------
GESTOS = {
    "siguiente": "right",  # mano abierta      -> flecha derecha
    "anterior": "left",    # puño cerrado      -> flecha izquierda
    "iniciar": "f5",       # dos dedos (V)     -> F5 inicia la presentación
    "finalizar": "esc",    # pulgar arriba     -> Esc termina la presentación
    "nada": None,          # mano relajada / cualquier otra pose -> no hace nada
}

DESCRIPCION = {
    "siguiente": "Mano abierta (5 dedos)",
    "anterior": "Puno cerrado",
    "iniciar": "Dos dedos (V)",
    "finalizar": "Pulgar arriba",
    "nada": "Mano relajada / otras poses",
}

# Imagen que ve la CNN: recorte cuadrado de la mano, en gris, TAM x TAM.
TAM_IMAGEN = 128
MARGEN_RECORTE = 0.30  # margen extra alrededor de la mano (fracción del lado)

# Cámara
CAMARA = 0
ANCHO_CAMARA = 640
ALTO_CAMARA = 480
