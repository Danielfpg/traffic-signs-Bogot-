# 🚦 Reconocimiento de Señales de Tránsito en Bogotá — CNN

**Proyecto académico | Análisis y Diseño de Algoritmos**  
Universidad Católica de Colombia  
Daniel Paredes · Jonathan Camacho · Prof. Juan Barrero

---

## Descripción

Sistema de reconocimiento automático de señales de tránsito mediante **Redes Neuronales Convolucionales (CNN)**, optimizado para condiciones adversas del entorno urbano de Bogotá.

### Condiciones adversas cubiertas

| # | Condición | Módulo responsable |
|---|-----------|-------------------|
| 1 | Noche / oscuridad total | `enhance_night()` en `image_enhancer.py` |
| 2 | Mala visión por oscuridad parcial | `enhance_low_light()` en `image_enhancer.py` |
| 3 | Condiciones generales de oscuridad | Pipeline adaptativo `adaptive_enhance()` |
| 4 | Señales con grafitis o daños por choques | `enhance_damaged_sign()` + augmentation en entrenamiento |
| 5 | Señales caídas o en posición inusual | `correct_rotation()` + `enhance_small_sign()` |

---

## Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────┐
│                     ENTRADA: Cámara / Video             │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  MÓDULO 1: Mejora Adaptativa de Imagen                  │
│  preprocessing/image_enhancer.py                        │
│                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────────┐ │
│  │ Detectar │──▶│ CLAHE en LAB │──▶│ Denoising +     │ │
│  │ nivel luz│   │ (noche/baja) │   │ Sharpening      │ │
│  └──────────┘   └──────────────┘   └─────────────────┘ │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  MÓDULO 2: Detección de Señales                         │
│  detection/sign_detector.py                             │
│                                                         │
│  Segmentación HSV (rojo/azul/amarillo)                  │
│  → Análisis de contornos y circularidad                 │
│  → Extracción de ROIs (bounding boxes)                  │
└─────────────────────┬───────────────────────────────────┘
                      │ ROIs detectadas
                      ▼
┌─────────────────────────────────────────────────────────┐
│  MÓDULO 2B: Mejora de ROI individual                    │
│                                                         │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ Top-hat morfo.  │  │ Upscaling    │  │ Corrección│  │
│  │ (grafitis/daño) │  │ (señal peq.) │  │ rotación  │  │
│  └─────────────────┘  └──────────────┘  └───────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  MÓDULO 3: Clasificador CNN                             │
│  classification/classifier.py                           │
│                                                         │
│  LeNet modificado (AridaneAM base)                      │
│  → Preprocesamiento con CLAHE (mejora sobre original)   │
│  → Predicción: 43 clases de señales                     │
│  → Umbral de confianza ajustable (default 60%)          │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  SALIDA: Frame anotado en tiempo real                   │
│  • Bounding boxes por señal detectada                   │
│  • Nombre de la señal + porcentaje de confianza         │
│  • Panel de estado: modo de iluminación + FPS           │
└─────────────────────────────────────────────────────────┘
```

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/Danielfpg/supervised-avanzado-german-traffic-signs.git
cd supervised-avanzado-german-traffic-signs

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar el dataset GTSRB (para reentrenamiento)
#    Descarga manual desde: https://benchmark.ini.rub.de/
#    o usa el dataset en pickle del repo AridaneAM:
#    https://bitbucket.org/jadslim/german-traffic-signs
```

---

## Uso

### Con cámara en tiempo real

```bash
# Cámara por defecto (índice 0) con modelo pre-entrenado
python main_camera.py --model modelo_v3.h5 --source 0

# Con ventana secundaria mostrando el frame mejorado
python main_camera.py --model modelo_v3.h5 --source 0 --enhanced
```

### Con video grabado

```bash
python main_camera.py --model modelo_v3.h5 --source video_bogota.mp4
```

### Con imagen estática (pruebas)

```bash
python main_camera.py --model modelo_v3.h5 --source señal.jpg --image
```

### Reentrenar el modelo

```bash
# Con el dataset GTSRB en formato pickle
python train_model.py --data german-traffic-signs/ --epochs 20 --output modelo_bogota_cnn.h5
```

**Controles durante ejecución:**
| Tecla | Acción |
|-------|--------|
| `q`   | Salir |
| `s`   | Guardar captura del frame actual |
| `e`   | Activar/desactivar ventana de frame mejorado |

---

## Mejoras sobre el repositorio base (AridaneAM)

| Componente | Original | Optimizado |
|------------|----------|------------|
| Ecualización | `equalizeHist` simple | **CLAHE** en espacio LAB |
| Nocturno | No soportado | Pipeline de mejora nocturna + denoising |
| Augmentation | Básico | + `brightness_range`, `channel_shift_range` |
| Detección | No (solo clasificación) | **Detector por color/forma** integrado |
| Señales dañadas | No | Filtro **Top-hat morfológico** |
| Señales caídas | No | **Corrección de rotación** por Hough |
| Señales pequeñas | No | **Upscaling** con INTER_CUBIC |
| Tiempo real | No (imágenes estáticas) | **Loop de cámara** completo |
| Callbacks de entrenamiento | No | ModelCheckpoint, EarlyStopping, ReduceLROnPlateau |

---

## Estructura del proyecto

```
traffic_signs_bogota/
├── main_camera.py              ← Pipeline principal (cámara en tiempo real)
├── train_model.py              ← Entrenamiento con augmentation mejorado
├── requirements.txt
├── preprocessing/
│   └── image_enhancer.py       ← Mejora adaptativa (noche, oscuridad, daños)
├── detection/
│   └── sign_detector.py        ← Detector por color/forma
├── classification/
│   └── classifier.py           ← CNN LeNet + preprocesamiento mejorado
└── models/
    └── (modelo_v3.h5 aquí)     ← Modelo pre-entrenado del repo base
```

---

## Dataset

- **Base:** [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/)
- **50,000+ imágenes, 43 clases**
- **Recomendación:** Complementar con imágenes capturadas en Bogotá para mejorar la adaptación local (señales colombianas, condiciones lumínicas locales)

---

## Referencias

- [AridaneAM/OpenCV-senales-de-trafico](https://github.com/AridaneAM/OpenCV-senales-de-trafico) — Base del clasificador
- [GTSRB Dataset](https://benchmark.ini.rub.de/) — Dataset de entrenamiento
- Concejo de Bogotá: 26,000 señales en mal estado (2024)
- Alcaldía de Bogotá: 12,000 siniestros viales en 2024, 30% entre 5–8pm
