"""
image_enhancer.py
=================
Módulo de mejora adaptativa de imagen para condiciones adversas.
Proyecto: Reconocimiento de Señales de Tránsito en Bogotá - CNN
Universidad Católica de Colombia

Condiciones cubiertas:
  1. Noche / oscuridad total
  2. Baja visibilidad por oscuridad parcial
  3. Condiciones generales de oscuridad
  4. Señales con grafitis o daños por choques
  5. Señales caídas o poco visibles por posición
"""

import cv2
import numpy as np


# ──────────────────────────────────────────────
# DETECCIÓN DE NIVEL DE LUZ
# ──────────────────────────────────────────────

def detect_light_level(frame: np.ndarray) -> float:
    """
    Retorna el brillo promedio de la imagen (0–255).
    Se usa para decidir qué pipeline de mejora aplicar.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def classify_lighting(frame: np.ndarray) -> str:
    """
    Clasifica la iluminación del frame:
      - 'night'    : < 40   (noche / oscuridad total)
      - 'low'      : 40–90  (baja visibilidad)
      - 'normal'   : > 90   (condiciones normales)
    """
    level = detect_light_level(frame)
    if level < 40:
        return 'night'
    elif level < 90:
        return 'low'
    return 'normal'


# ──────────────────────────────────────────────
# PIPELINE 1: MEJORA NOCTURNA
# Condiciones 1, 2 y 3
# ──────────────────────────────────────────────

def enhance_night(frame: np.ndarray) -> np.ndarray:
    """
    Pipeline agresivo para imágenes muy oscuras.
    Trabaja en espacio LAB para preservar color al amplificar luminosidad.
    """
    # 1. Convertir a LAB y amplificar canal L
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE con clip alto para escenas muy oscuras
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Estirar histograma al rango completo
    l_enhanced = cv2.normalize(l_enhanced, None, 0, 255, cv2.NORM_MINMAX)

    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # 2. Reducción de ruido (el ruido aumenta mucho en ISO alto / baja luz)
    denoised = cv2.fastNlMeansDenoisingColored(enhanced_bgr, None,
                                                h=10, hColor=10,
                                                templateWindowSize=7,
                                                searchWindowSize=21)

    # 3. Leve sharpening para recuperar bordes
    kernel_sharp = np.array([[0, -1,  0],
                              [-1,  5, -1],
                              [0, -1,  0]])
    sharpened = cv2.filter2D(denoised, -1, kernel_sharp)

    return sharpened


def enhance_low_light(frame: np.ndarray) -> np.ndarray:
    """
    Pipeline moderado para baja visibilidad (no noche total).
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    enhanced = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Denoising suave
    result = cv2.bilateralFilter(result, d=9, sigmaColor=75, sigmaSpace=75)

    return result


# ──────────────────────────────────────────────
# PIPELINE 2: SEÑALES DAÑADAS / GRAFITIS
# Condición 4
# ──────────────────────────────────────────────

def enhance_damaged_sign(roi: np.ndarray) -> np.ndarray:
    """
    Mejora una ROI (región de interés) de una señal posiblemente dañada.
    Aplica ecualización y filtrado morfológico para resaltar
    la estructura geométrica de la señal bajo grafitis o desgaste.
    """
    # Convertir a escala de grises para análisis
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # CLAHE para contrastar el contenido real de la señal
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced_gray = clahe.apply(gray)

    # Top-hat morfológico: resalta estructuras pequeñas (texto, bordes)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(enhanced_gray, cv2.MORPH_TOPHAT, kernel)

    # Combinar con imagen mejorada
    combined = cv2.addWeighted(enhanced_gray, 0.8, tophat, 0.2, 0)

    # Volver a BGR para mantener compatibilidad con el resto del pipeline
    result_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)

    return result_bgr


# ──────────────────────────────────────────────
# PIPELINE 3: SEÑALES EN POSICIÓN INUSUAL / PEQUEÑAS
# Condición 5
# ──────────────────────────────────────────────

def enhance_small_sign(roi: np.ndarray, target_size: int = 64) -> np.ndarray:
    """
    Upscaling inteligente para señales muy pequeñas o mal posicionadas.
    Usa INTER_CUBIC para preservar bordes al ampliar.
    """
    h, w = roi.shape[:2]
    scale = target_size / max(h, w)

    if scale > 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        upscaled = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Sharpening post-upscale
        kernel_sharp = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
        return cv2.filter2D(upscaled, -1, kernel_sharp)

    return roi


def correct_rotation(roi: np.ndarray) -> np.ndarray:
    """
    Intenta detectar y corregir rotaciones moderadas en señales caídas
    usando la transformada de Hough sobre bordes.
    Útil para señales volcadas o parcialmente caídas.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=30, minLineLength=20, maxLineGap=10)

    if lines is None:
        return roi

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)

    # Solo corregir si la inclinación es significativa (> 5°) pero manejable (< 45°)
    if 5 < abs(median_angle) < 45:
        h, w = roi.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        corrected = cv2.warpAffine(roi, M, (w, h),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
        return corrected

    return roi


# ──────────────────────────────────────────────
# PIPELINE MAESTRO ADAPTATIVO
# ──────────────────────────────────────────────

def adaptive_enhance(frame: np.ndarray, roi: np.ndarray = None) -> tuple:
    """
    Pipeline maestro que combina todos los módulos según las condiciones
    detectadas automáticamente.

    Parámetros:
        frame : Frame completo de la cámara
        roi   : Región de interés (recorte de señal), opcional

    Retorna:
        (frame_mejorado, roi_mejorada, lighting_mode)
    """
    lighting = classify_lighting(frame)

    # Mejorar el frame completo según iluminación
    if lighting == 'night':
        enhanced_frame = enhance_night(frame)
    elif lighting == 'low':
        enhanced_frame = enhance_low_light(frame)
    else:
        enhanced_frame = frame.copy()

    # Mejorar la ROI si existe
    enhanced_roi = None
    if roi is not None:
        # Condición 4: mejorar señal potencialmente dañada
        enhanced_roi = enhance_damaged_sign(roi)

        # Condición 5: si la señal es muy pequeña, ampliarla
        h, w = roi.shape[:2]
        if max(h, w) < 48:
            enhanced_roi = enhance_small_sign(enhanced_roi)

        # Condición 5: corregir posición si está inclinada
        enhanced_roi = correct_rotation(enhanced_roi)

    return enhanced_frame, enhanced_roi, lighting
