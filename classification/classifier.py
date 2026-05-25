"""
classifier.py
=============
Clasificador CNN de señales de tránsito.
Basado en la arquitectura LeNet del repositorio AridaneAM/OpenCV-senales-de-trafico
con mejoras de preprocesamiento para condiciones adversas.

Proyecto: Reconocimiento de Señales de Tránsito en Bogotá - CNN
Universidad Católica de Colombia
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import cv2
import numpy as np

# Intento de carga de Keras/TensorFlow con manejo de versiones


try:
    import tf_keras
    from tf_keras.models import load_model, Sequential
    from tf_keras.layers import (Conv2D, MaxPooling2D,
                                  Flatten, Dense, Dropout)
    from tf_keras.optimizers import Adam
    from tf_keras.utils import to_categorical
    from tf_keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[ADVERTENCIA] TensorFlow no disponible. Solo modo de demostración.")
except Exception as e:          # <-- Exception en vez de ImportError
    TF_AVAILABLE = False
    print(f"[ADVERTENCIA] Error al importar: {e}")

# ──────────────────────────────────────────────
# ETIQUETAS DE LAS 43 CLASES (GTSRB)
# ──────────────────────────────────────────────

CLASS_NAMES = {
    0:  "Límite velocidad 20 km/h",
    1:  "Límite velocidad 30 km/h",
    2:  "Límite velocidad 50 km/h",
    3:  "Límite velocidad 60 km/h",
    4:  "Límite velocidad 70 km/h",
    5:  "Límite velocidad 80 km/h",
    6:  "Fin límite 80 km/h",
    7:  "Límite velocidad 100 km/h",
    8:  "Límite velocidad 120 km/h",
    9:  "Prohibido adelantar",
    10: "Prohibido adelantar (camiones)",
    11: "Cruce con prioridad",
    12: "Vía con prioridad",
    13: "Ceda el paso",
    14: "STOP",
    15: "Prohibido circular",
    16: "Prohibido camiones",
    17: "Prohibida la entrada",
    18: "Precaución general",
    19: "Curva peligrosa izquierda",
    20: "Curva peligrosa derecha",
    21: "Curva doble",
    22: "Pavimento irregular",
    23: "Pavimento deslizante",
    24: "Estrechamiento derecha",
    25: "Obras en la vía",
    26: "Semáforos",
    27: "Peatones",
    28: "Zona escolar",
    29: "Ciclistas",
    30: "Hielo/nieve",
    31: "Animales en la vía",
    32: "Fin restricciones",
    33: "Gire a la derecha",
    34: "Gire a la izquierda",
    35: "Siga recto",
    36: "Recto o derecha",
    37: "Recto o izquierda",
    38: "Mantenga la derecha",
    39: "Mantenga la izquierda",
    40: "Rotonda obligatoria",
    41: "Fin prohibición adelantar",
    42: "Fin prohibición adelantar (camiones)",
}


# ──────────────────────────────────────────────
# PREPROCESAMIENTO MEJORADO
# Reemplaza el grayscale+equalizeHist simple del repo original
# ──────────────────────────────────────────────

def preprocess_for_model(image: np.ndarray) -> np.ndarray:
    """
    Preprocesa una imagen para ingresarla al modelo CNN.
    Mejora sobre el repo original: usa CLAHE en vez de equalizeHist simple.

    Pipeline:
        BGR → Escala de grises → CLAHE → Normalización → reshape
    """
    # Redimensionar a 32x32 (tamaño de entrada del modelo)
    img = cv2.resize(image, (32, 32))

    # Escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE en lugar de equalizeHist: mejor para condiciones variables
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    equalized = clahe.apply(gray)

    # Normalizar 0-1
    normalized = equalized / 255.0

    # Reshape para Keras: (1, 32, 32, 1)
    return normalized.reshape(1, 32, 32, 1)


# ──────────────────────────────────────────────
# ARQUITECTURA DEL MODELO (LeNet modificado)
# ──────────────────────────────────────────────

def build_model(num_classes: int = 43) -> "Sequential":
    """
    Construye la arquitectura CNN modificada del repo AridaneAM.
    Dos bloques Conv → Conv → MaxPool + capa densa con Dropout.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow no está instalado.")

    model = Sequential([
        Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'),
        Conv2D(60, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(30, (3, 3), activation='relu'),
        Conv2D(30, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(500, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ──────────────────────────────────────────────
# GENERADOR DE DATOS CON AUGMENTATION PARA CONDICIONES ADVERSAS
# ──────────────────────────────────────────────

def build_data_generator() -> "ImageDataGenerator":
    """
    ImageDataGenerator con augmentation específico para condiciones adversas.
    Cubre:
      - Oscuridad variable (brightness_range)
      - Señales ligeramente giradas / caídas (rotation_range)
      - Señales en posiciones inusuales (shift, zoom)
      - Desgaste y variación de color (channel_shift_range)
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow no está instalado.")

    return ImageDataGenerator(
        width_shift_range=0.15,          # desplazamiento horizontal
        height_shift_range=0.15,         # desplazamiento vertical
        zoom_range=0.3,                  # zoom (señales lejanas/cercanas)
        shear_range=0.15,                # deformación (perspectiva)
        rotation_range=20,               # rotación (señales caídas)
        brightness_range=[0.2, 1.5],     # CLAVE: simula noche y sobreexposición
        channel_shift_range=30.0,        # simula suciedad, grafitis, decoloración
        fill_mode='nearest'
    )


# ──────────────────────────────────────────────
# CLASIFICADOR
# ──────────────────────────────────────────────

class TrafficSignClassifier:
    """
    Clasificador de señales de tránsito con soporte para condiciones adversas.
    """

    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def load(self, path: str):
        """Carga un modelo .h5 previamente entrenado."""
        if not TF_AVAILABLE:
            print("[INFO] Modo demo: TensorFlow no disponible.")
            return
        self.model = load_model(path)
        print(f"[INFO] Modelo cargado desde: {path}")

    def predict(self, roi: np.ndarray, confidence_threshold: float = 0.60) -> dict:
        """
        Clasifica una región de interés (señal recortada).

        Retorna un diccionario con:
            class_id    : ID de la clase predicha
            class_name  : Nombre legible de la señal
            confidence  : Probabilidad (0–1)
            valid       : True si supera el umbral de confianza
        """
        if self.model is None:
            return {
                'class_id': -1,
                'class_name': 'Modelo no cargado',
                'confidence': 0.0,
                'valid': False
            }

        processed = preprocess_for_model(roi)
        predictions = self.model.predict(processed, verbose=0)[0]

        class_id = int(np.argmax(predictions))
        confidence = float(predictions[class_id])
        class_name = CLASS_NAMES.get(class_id, f"Clase {class_id}")

        return {
            'class_id': class_id,
            'class_name': class_name,
            'confidence': confidence,
            'valid': confidence >= confidence_threshold
        }
