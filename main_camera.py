"""
main_camera.py
==============
Pipeline principal de detección y clasificación de señales de tránsito
en tiempo real desde cámara.

Integra:
  1. Mejora adaptativa de imagen (image_enhancer.py)
  2. Detección de señales por color/forma (sign_detector.py)
  3. Clasificación CNN (classifier.py)

Condiciones cubiertas:
  ✅ 1. Noche
  ✅ 2. Mala visión por oscuridad
  ✅ 3. Condiciones de oscuridad
  ✅ 4. Señales con grafitis o daños
  ✅ 5. Señales caídas o en posiciones inusuales

Uso:
    python main_camera.py --model modelo_v3.h5 --source 0
    python main_camera.py --model modelo_v3.h5 --source video.mp4
    python main_camera.py --model modelo_v3.h5 --source imagen.jpg --image

Proyecto: Reconocimiento de Señales de Tránsito en Bogotá - CNN
Universidad Católica de Colombia - Daniel Paredes / Jonathan Camacho
"""

import cv2
import numpy as np
import argparse
import sys
import os
import time

# Agregar el directorio raíz al path para importar los módulos
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from preprocessing.image_enhancer import adaptive_enhance, classify_lighting
from detection.sign_detector import SignDetector
from classification.classifier import TrafficSignClassifier


# ──────────────────────────────────────────────
# OVERLAY DE INFORMACIÓN EN PANTALLA
# ──────────────────────────────────────────────

def draw_status_overlay(frame: np.ndarray, lighting: str,
                         fps: float, n_detections: int) -> np.ndarray:
    """Dibuja el panel de estado en la esquina superior izquierda."""
    overlay = frame.copy()

    # Fondo semitransparente
    cv2.rectangle(overlay, (0, 0), (320, 100), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Modos de iluminación con color indicativo
    mode_colors = {
        'night': (0,   80,  255),   # rojo oscuro → peligro
        'low':   (0,   200, 255),   # amarillo → precaución
        'normal':(0,   220, 100),   # verde → OK
    }
    mode_labels = {
        'night':  '🌑 MODO NOCTURNO',
        'low':    '🌆 BAJA VISIBILIDAD',
        'normal': '☀️  VISIBILIDAD NORMAL',
    }

    color = mode_colors.get(lighting, (200, 200, 200))
    label = mode_labels.get(lighting, lighting)

    cv2.putText(frame, label,       (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Señales detectadas: {n_detections}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    return frame


def draw_result_panel(frame: np.ndarray, results: list) -> np.ndarray:
    """Dibuja un panel lateral con los resultados de clasificación."""
    if not results:
        return frame

    h, w = frame.shape[:2]
    panel_w = 300
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    cv2.putText(panel, "CLASIFICACION", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.line(panel, (10, 40), (panel_w - 10, 40), (100, 100, 100), 1)

    y_offset = 60
    for i, res in enumerate(results):
        if not res['valid']:
            continue

        # Barra de confianza
        conf_pct = int(res['confidence'] * (panel_w - 30))
        bar_color = (0, 200, 100) if res['confidence'] > 0.80 else (0, 165, 255)
        cv2.rectangle(panel, (15, y_offset),
                      (15 + conf_pct, y_offset + 12), bar_color, -1)

        # Texto
        name = res['class_name']
        if len(name) > 28:
            name = name[:25] + "..."
        cv2.putText(panel, f"{i+1}. {name}", (10, y_offset + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        cv2.putText(panel, f"   {res['confidence']*100:.1f}%", (10, y_offset + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, bar_color, 1)

        y_offset += 65
        if y_offset > h - 40:
            break

    return np.hstack([frame, panel])


# ──────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────

def run_camera(model_path: str, source, show_enhanced: bool = False):
    """
    Loop principal de captura y procesamiento.
    """
    # Inicializar componentes
    detector   = SignDetector(min_area=600, max_area=100_000, max_detections=5)
    classifier = TrafficSignClassifier(model_path=model_path)

    # Abrir fuente de video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la fuente: {source}")
        sys.exit(1)

    print("[INFO] Sistema iniciado. Presiona 'q' para salir, 's' para captura.")
    print(f"[INFO] Modelo: {model_path if model_path else 'No cargado (solo detección)'}")

    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # Si es video, reiniciar al final
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame_count += 1

        # ── PASO 1: Mejora adaptativa según condición de iluminación ──
        enhanced_frame, _, lighting = adaptive_enhance(frame)

        # ── PASO 2: Detección de señales ──
        detections = detector.detect(enhanced_frame)

        # ── PASO 3: Clasificación de cada señal detectada ──
        results = []
        for det in detections:
            # Mejorar la ROI específicamente para la señal
            _, enhanced_roi, _ = adaptive_enhance(frame, roi=det.roi)
            roi_to_classify = enhanced_roi if enhanced_roi is not None else det.roi

            result = classifier.predict(roi_to_classify, confidence_threshold=0.60)
            results.append(result)

        # ── PASO 4: Visualización ──
        output = detector.draw_detections(enhanced_frame, detections, results)

        # Calcular FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        # Panel de estado
        output = draw_status_overlay(output, lighting, fps, len(detections))

        # Panel lateral de resultados
        output = draw_result_panel(output, results)

        # Mostrar frame mejorado en ventana secundaria (opcional)
        if show_enhanced:
            cv2.imshow("Frame Mejorado", enhanced_frame)

        cv2.imshow("Reconocimiento de Señales - Bogotá CNN", output)

        # Controles de teclado
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Saliendo...")
            break
        elif key == ord('s'):
            filename = f"captura_{frame_count}.jpg"
            cv2.imwrite(filename, output)
            print(f"[INFO] Captura guardada: {filename}")
        elif key == ord('e'):
            show_enhanced = not show_enhanced

    cap.release()
    cv2.destroyAllWindows()


def run_image(image_path: str, model_path: str):
    """
    Procesa una sola imagen estática.
    Útil para pruebas y validación.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] No se pudo cargar: {image_path}")
        sys.exit(1)

    detector   = SignDetector(min_area=400, max_area=100_000)
    classifier = TrafficSignClassifier(model_path=model_path)

    # Mejora adaptativa
    enhanced_frame, _, lighting = adaptive_enhance(frame)
    print(f"[INFO] Condición de iluminación detectada: {lighting}")

    # Detección
    detections = detector.detect(enhanced_frame)
    print(f"[INFO] Señales detectadas: {len(detections)}")

    # Clasificación
    results = []
    for i, det in enumerate(detections):
        _, enhanced_roi, _ = adaptive_enhance(frame, roi=det.roi)
        roi_to_classify = enhanced_roi if enhanced_roi is not None else det.roi
        result = classifier.predict(roi_to_classify)
        results.append(result)
        print(f"  [{i+1}] {result['class_name']} — confianza: {result['confidence']*100:.1f}%")

    # Visualización
    output = detector.draw_detections(enhanced_frame, detections, results)
    output = draw_result_panel(output, results)

    cv2.imshow("Resultado", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconocimiento de Señales de Tránsito - Bogotá CNN"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Ruta al modelo .h5 (ej: modelo_v3.h5)"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Fuente de video: 0=webcam, ruta a video o imagen"
    )
    parser.add_argument(
        "--image", action="store_true",
        help="Procesar una imagen estática en vez de video"
    )
    parser.add_argument(
        "--enhanced", action="store_true",
        help="Mostrar ventana secundaria con frame mejorado"
    )

    args = parser.parse_args()

    # Convertir source a int si es número (índice de cámara)
    source = int(args.source) if args.source.isdigit() else args.source

    if args.image:
        run_image(args.source, args.model)
    else:
        run_camera(args.model, source, show_enhanced=args.enhanced)
