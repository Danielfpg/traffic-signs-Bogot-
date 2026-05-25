"""
train_model.py
==============
Entrenamiento del modelo CNN con augmentation específico
para condiciones adversas en Bogotá.

Basado en: AridaneAM/OpenCV-senales-de-trafico
Mejoras:
  - CLAHE en lugar de equalizeHist
  - brightness_range para simular noche/oscuridad
  - channel_shift_range para simular grafitis/deterioro
  - rotation_range aumentado para señales caídas
  - Más épocas y callbacks de mejora

Uso:
    python train_model.py --data german-traffic-signs/ --epochs 20 --output mi_modelo.h5

Proyecto: Reconocimiento de Señales de Tránsito en Bogotá - CNN
Universidad Católica de Colombia
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import pickle
import argparse
import cv2



try:
    from tf_keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                    ReduceLROnPlateau)
    from tf_keras.utils import to_categorical
    import matplotlib.pyplot as plt

    TF_OK = True
except ImportError:
    TF_OK = False
    print("[ERROR] TensorFlow no está instalado. Instala con: pip install tensorflow")

except Exception  as e:
    TF_OK = False
    print(f"[ERROR] Import falló: {e}")

# Importar desde el módulo del proyecto
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classification.classifier import build_model, build_data_generator, preprocess_for_model


# ──────────────────────────────────────────────
# PREPROCESAMIENTO MEJORADO (con CLAHE)
# ──────────────────────────────────────────────

def preprocess_batch(images: np.ndarray) -> np.ndarray:
    """
    Aplica preprocesamiento mejorado a un batch de imágenes.
    Reemplaza el grayscale+equalizeHist simple del repo original.
    """
    processed = []
    for img in images:
        # Convertir de RGB a BGR si viene del pickle (que usa BGR)
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # CLAHE en lugar de equalizeHist
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        equalized = clahe.apply(gray)

        normalized = equalized / 255.0
        processed.append(normalized)

    result = np.array(processed)
    return result.reshape(-1, 32, 32, 1)


# ──────────────────────────────────────────────
# CARGA DE DATOS
# ──────────────────────────────────────────────

def load_gtsrb_data(data_dir: str) -> tuple:
    """Carga el dataset GTSRB desde archivos pickle."""
    sets = {}
    for split in ['train', 'valid', 'test']:
        path = os.path.join(data_dir, f'{split}.p')
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró: {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        sets[split] = (data['features'], data['labels'])
        print(f"[INFO] {split}: {data['features'].shape[0]} imágenes")

    return sets['train'], sets['valid'], sets['test']


# ──────────────────────────────────────────────
# VISUALIZACIÓN DE MÉTRICAS
# ──────────────────────────────────────────────

def plot_training_history(history, output_path: str = "training_history.png"):
    """Guarda gráficas de accuracy y loss del entrenamiento."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['loss'],     label='Entrenamiento')
    axes[0].plot(history.history['val_loss'], label='Validación')
    axes[0].set_title('Pérdida por época')
    axes[0].set_xlabel('Épocas')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history['accuracy'],     label='Entrenamiento')
    axes[1].plot(history.history['val_accuracy'], label='Validación')
    axes[1].set_title('Precisión por época')
    axes[1].set_xlabel('Épocas')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[INFO] Gráficas guardadas en: {output_path}")


# ──────────────────────────────────────────────
# ENTRENAMIENTO PRINCIPAL
# ──────────────────────────────────────────────

def train(data_dir: str, epochs: int, output_path: str,
          batch_size: int = 50):

    if not TF_OK:
        print("[ERROR] TensorFlow requerido para entrenar.")
        return

    print("\n[INFO] ══════════════════════════════════════")
    print("[INFO]  Entrenamiento CNN - Señales Bogotá")
    print("[INFO] ══════════════════════════════════════\n")

    # 1. Cargar datos
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_gtsrb_data(data_dir)

    # 2. Preprocesamiento con CLAHE mejorado
    print("[INFO] Aplicando preprocesamiento mejorado (CLAHE)...")
    X_train = preprocess_batch(X_train)
    X_val   = preprocess_batch(X_val)
    X_test  = preprocess_batch(X_test)

    # 3. One-hot encoding
    num_classes = 43
    y_train = to_categorical(y_train, num_classes)
    y_val   = to_categorical(y_val,   num_classes)
    y_test  = to_categorical(y_test,  num_classes)

    print(f"[INFO] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 4. Generador con augmentation para condiciones adversas
    datagen = build_data_generator()
    datagen.fit(X_train)

    # 5. Construir modelo
    model = build_model(num_classes)
    model.summary()

    # 6. Callbacks
    callbacks = [
        # Guardar el mejor modelo automáticamente
        ModelCheckpoint(
            filepath=output_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Detener si no mejora en 5 épocas
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Reducir LR si el val_loss no mejora
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    # 7. Entrenamiento
    print(f"\n[INFO] Iniciando entrenamiento: {epochs} épocas, batch_size={batch_size}")
    steps = max(1, len(X_train) // batch_size)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=steps,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # 8. Evaluación final
    print("\n[INFO] Evaluando en test set...")
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULTADO] Loss:     {score[0]:.4f}")
    print(f"[RESULTADO] Accuracy: {score[1]*100:.2f}%")

    # 9. Guardar gráficas
    plot_training_history(history)

    print(f"\n[INFO] Modelo guardado en: {output_path}")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento CNN - Señales de Tránsito Bogotá"
    )
    parser.add_argument("--data",    type=str, default="german-traffic-signs/",
                        help="Directorio con train.p, valid.p, test.p")
    parser.add_argument("--epochs",  type=int, default=20,
                        help="Número de épocas de entrenamiento")
    parser.add_argument("--output",  type=str, default="modelo_bogota_cnn.h5",
                        help="Ruta de salida del modelo .h5")
    parser.add_argument("--batch",   type=int, default=50,
                        help="Tamaño del batch")

    args = parser.parse_args()
    train(args.data, args.epochs, args.output, args.batch)
