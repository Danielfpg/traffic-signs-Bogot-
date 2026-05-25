"""
sign_detector.py
================
Detector de señales de tránsito por análisis de color y forma.
Localiza las regiones de la imagen donde probablemente hay una señal,
extrae el recorte (ROI) y lo pasa al clasificador.

Detecta señales por:
  - Segmentación de color (rojo, azul, amarillo) en espacio HSV
  - Análisis de contornos y forma (circularidad, relación de aspecto)
  - Filtros de área mínima y máxima

Compatible con:
  - Condición 5: señales caídas (no exige posición vertical)
  - Condición 4: señales dañadas (umbral de color flexible)
  - Condición 1-3: mejorado por el pipeline del image_enhancer antes de llamarse

Proyecto: Reconocimiento de Señales de Tránsito en Bogotá - CNN
Universidad Católica de Colombia
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Detection:
    """Representa una señal detectada en el frame."""
    bbox: tuple          # (x, y, w, h)
    roi: np.ndarray      # recorte de la señal
    area: float          # área del contorno
    circularity: float   # qué tan circular es (1.0 = círculo perfecto)
    color_type: str      # 'red', 'blue', 'yellow', 'unknown'


# ──────────────────────────────────────────────
# RANGOS DE COLOR EN HSV
# ──────────────────────────────────────────────

# Rojo (aparece en dos rangos en HSV)
RED_LOWER_1  = np.array([0,   70,  50])
RED_UPPER_1  = np.array([10, 255, 255])
RED_LOWER_2  = np.array([160, 70,  50])
RED_UPPER_2  = np.array([180, 255, 255])

# Azul (señales informativas y obligatorias)
BLUE_LOWER   = np.array([100, 80,  50])
BLUE_UPPER   = np.array([130, 255, 255])

# Amarillo (señales de precaución/obras)
YELLOW_LOWER = np.array([20,  80,  80])
YELLOW_UPPER = np.array([35, 255, 255])


def _color_mask(hsv: np.ndarray) -> tuple:
    """
    Crea máscaras de color para detectar señales.
    Retorna (máscara combinada, tipo de color dominante).
    """
    red_mask = (
        cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1) |
        cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    )
    blue_mask   = cv2.inRange(hsv, BLUE_LOWER,   BLUE_UPPER)
    yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)

    # Limpiar ruido morfológico
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for mask in [red_mask, blue_mask, yellow_mask]:
        cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, dst=mask)
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, dst=mask)

    combined = red_mask | blue_mask | yellow_mask

    # Determinar color dominante
    counts = {
        'red':    cv2.countNonZero(red_mask),
        'blue':   cv2.countNonZero(blue_mask),
        'yellow': cv2.countNonZero(yellow_mask),
    }
    dominant = max(counts, key=counts.get)
    if counts[dominant] == 0:
        dominant = 'unknown'

    return combined, dominant, red_mask, blue_mask, yellow_mask


def _contour_to_detection(contour, frame: np.ndarray,
                           color_type: str,
                           min_area: int, max_area: int) -> Detection | None:
    """
    Convierte un contorno en una Detection si cumple los criterios de forma.
    """
    area = cv2.contourArea(contour)
    if not (min_area <= area <= max_area):
        return None

    # Circularidad: 4π·área / perímetro²  → 1.0 es círculo perfecto
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None
    circularity = 4 * np.pi * area / (perimeter ** 2)

    # Aceptar formas entre 0.4 (triángulo/rectángulo) y 1.2 (círculo)
    # Umbral bajo para cubrir señales dañadas o inclinadas (cond. 4 y 5)
    if circularity < 0.35:
        return None

    x, y, w, h = cv2.boundingRect(contour)

    # Relación de aspecto: señales válidas son aproximadamente cuadradas
    aspect = w / h if h > 0 else 0
    if not (0.4 <= aspect <= 2.5):
        return None

    # Margen pequeño para capturar bordes completos
    margin = 8
    h_img, w_img = frame.shape[:2]
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w_img, x + w + margin)
    y2 = min(h_img, y + h + margin)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    return Detection(
        bbox=(x1, y1, x2 - x1, y2 - y1),
        roi=roi,
        area=area,
        circularity=circularity,
        color_type=color_type
    )


# ──────────────────────────────────────────────
# DETECTOR PRINCIPAL
# ──────────────────────────────────────────────

class SignDetector:
    """
    Detecta señales de tránsito en un frame usando segmentación de color.
    """

    def __init__(self,
                 min_area: int = 800,
                 max_area: int = 80_000,
                 max_detections: int = 5):
        """
        Parámetros:
            min_area        : Área mínima de contorno para considerar como señal
            max_area        : Área máxima (evita falsos positivos grandes)
            max_detections  : Máximo de señales a retornar por frame
        """
        self.min_area = min_area
        self.max_area = max_area
        self.max_detections = max_detections

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detecta señales en el frame.

        Retorna lista de Detection ordenada por área (mayor primero).
        """
        # Suavizar para reducir ruido antes de segmentar
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        combined_mask, _, red_mask, blue_mask, yellow_mask = _color_mask(hsv)

        # Mapa color → máscara para etiquetar cada contorno
        color_masks = {
            'red':    red_mask,
            'blue':   blue_mask,
            'yellow': yellow_mask,
        }

        detections = []

        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            # Determinar el color dominante para este contorno específico
            x, y, w, h = cv2.boundingRect(contour)
            best_color = 'unknown'
            best_count = 0
            for color_name, mask in color_masks.items():
                roi_mask = mask[y:y+h, x:x+w]
                count = cv2.countNonZero(roi_mask)
                if count > best_count:
                    best_count = count
                    best_color = color_name

            det = _contour_to_detection(
                contour, frame, best_color,
                self.min_area, self.max_area
            )
            if det is not None:
                detections.append(det)

        # Ordenar por área (señales más grandes = más cercanas = más prioritarias)
        detections.sort(key=lambda d: d.area, reverse=True)

        return detections[:self.max_detections]

    def draw_detections(self, frame: np.ndarray,
                        detections: List[Detection],
                        results: list = None) -> np.ndarray:
        """
        Dibuja las detecciones sobre el frame.

        Parámetros:
            frame      : Frame original
            detections : Lista de Detection
            results    : Lista de dicts retornados por el clasificador (opcional)
        """
        output = frame.copy()

        color_map = {
            'red':    (0,   0,   255),
            'blue':   (255, 0,   0),
            'yellow': (0,   200, 255),
            'unknown':(128, 128, 128),
        }

        for i, det in enumerate(detections):
            x, y, w, h = det.bbox
            color = color_map.get(det.color_type, (200, 200, 200))

            # Bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

            # Etiqueta del clasificador (si está disponible)
            if results and i < len(results) and results[i]['valid']:
                label = f"{results[i]['class_name']} {results[i]['confidence']*100:.0f}%"
            else:
                label = f"Señal ({det.color_type})"

            # Fondo para el texto
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x, y - th - 8), (x + tw + 4, y), color, -1)
            cv2.putText(output, label, (x + 2, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)

        return output
