# Trabajo práctico final de Procesamiento Digital de Imágenes

## Objetivo y alcance

Localizar una patente por fotografía estática de un automóvil argentino y comparar dos métodos de reconocimiento sobre las mismas imágenes. Los formatos son `ABC123` y `AB123CD`. Las etapas 1 a 5 cubren configuración, OCR básico, preprocesamiento, detección, diagnóstico y segmentación individual. Hay 58 fotos verificadas de desarrollo y 25 reservadas para evaluación final. La etapa 6 requiere una solicitud explícita.

## Restricciones técnicas

- Usar Python, OpenCV, NumPy, Matplotlib, Pandas, Pillow, PyTesseract, Jupyter y Pytest según corresponda a cada etapa. Usar rutas relativas al proyecto con `pathlib`.
- Implementar la detección con técnicas clásicas de procesamiento digital. El alcance excluye YOLO, detectores preentrenados, OCR remoto, video, interfaz gráfica, aplicación web, API y base de datos.
- Implementar la BPN y su retropropagación manual con NumPy. Tesseract es el motor externo del método A. Los dos métodos deben poder ejecutarse y evaluarse de forma independiente.
- Conservar el texto original de cada método. La normalización se limita al alfabeto permitido. Si más adelante se estudian correcciones basadas en el formato, registrar la versión corregida en un campo aparte y conservar la predicción original.
- Mantener explicaciones, resultados y visualizaciones en español en `notebooks/TP_Final_OCR.ipynb`. Mostrar entradas y etapas intermedias a medida que estén implementadas. El notebook debe ejecutar con un kernel nuevo y cargar pesos externos cuando corresponda.

## Referencias del profesor

`references/` es de solo lectura para el desarrollo. Adaptar la lógica en `src/`.

- `bpn-c-code-student-guide.pdf` explica la red 100 → 16 → 10 y la actualización por patrón.
- `bpn-tutorial-v16.pdf` deriva el error cuadrático, los deltas y el descenso por gradiente.
- `bpn_mnist_V5_trained.py` implementa 784 → 64 → 10, propagación vectorizada, sigmoide, retropropagación por mini-batches y persistencia NPZ. TensorFlow/Keras aparece en el original como cargador de MNIST.
- `dig2.png` muestra un 2 manuscrito para probar el reconocimiento externo del script de referencia.

La BPN del TP parte de 784 → 64 → 36 y del orden de clases `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`. Usar caracteres claros sobre fondo negro, 28 × 28, proporción conservada y centrado por centro de masa cuando corresponda. Normalizar a `[0, 1]` y aplanar a 784 componentes. Configurar la semilla, cantidad de neuronas ocultas, tasa de aprendizaje, tamaño de lote y épocas. Conservar sigmoide y su derivada, objetivos one-hot, error cuadrático, argmax y pesos NPZ. Entrenar pesos propios y tratar las activaciones como puntajes.

## Arquitectura

| Ruta | Responsabilidad |
| --- | --- |
| `data/raw/roboflow_argentina/` | Export Pascal VOC de Roboflow Master v1, conservado como fuente local. |
| `data/original/`, `data/cropped/`, `data/ground_truth.csv` | Copias de fotografías, recortes de referencia y etiquetas humanas con partición experimental. |
| `data/examples/`, `data/characters/`, `data/synthetic/` | Pruebas funcionales, caracteres segmentados y futuro dataset sintético de entrenamiento. |
| `data/character_annotations.csv` | Cajas de caracteres dibujadas por una persona sobre el recorte XML original. |
| `src/ocr.py` | Carga, reconocimiento con PyTesseract y validación de formato. |
| `src/ocr_preparation.py` | Escala, polaridad y preparación de una región OCR completa en la etapa 4B. |
| `src/preprocessing.py`, `src/detection.py`, `src/segmentation.py` | Preprocesamiento, localización y extracción de caracteres. |
| `src/dataset.py`, `src/neural_network.py` | Preparación del conjunto OCR y futura clasificación neuronal. |
| `src/evaluation.py`, `src/pipeline.py` | Evaluación M0–M9 y futura integración de las ramas. |
| `scripts/` | Importación Pascal VOC, anotación humana y comparaciones reproducibles de preprocesamiento, detección y segmentación. |
| `models/`, `results/` | Pesos, metadatos, detecciones, imágenes procesadas, caracteres y gráficos. |
| `train.py`, `main.py` | Entrenamiento y procesamiento por lotes con exportación a TXT y CSV. |
| `notebooks/`, `tests/` | Presentación académica y pruebas. |

## Etapas

1. Configuración inicial y análisis de referencias.
2. OCR básico sobre recortes, con PSM 7 y 8.
3. Importación de recortes de referencia, etiquetado y preprocesamiento espacial M0–M9 con PSM fijo.
4. Detección por bordes, morfología, contornos y criterios geométricos. Extracción y corrección de perspectiva cuando corresponda.
   - Etapa 4B. Auditoría de insumos, ensayo incremental OCR, diagnóstico de candidatos y comparación de cuatro pipelines.
5. Segmentación y normalización de caracteres.
6. Dataset balanceado de 36 clases. Documentar procedencia y licencia de tipografías, aplicar transformaciones de manera controlada y dividir por grupos de ejemplos base para evitar fugas.
7. BPN vectorizada basada en la referencia.
8. Entrenamiento reproducible y evaluación en validación, prueba y caracteres reales independientes.
9. Integración y procesamiento por lotes.
10. Comparación experimental, documentación y conclusiones.

Distinguir las evaluaciones sobre recortes y las de extremo a extremo. Contabilizar fallos de detección en el conjunto evaluado. Medir localización, acierto de carácter, acierto de patente, CER y tiempo. Registrar matriz de confusión y curvas de error y acierto del entrenamiento y validación. Informar resultados observados y detenerse después de la etapa autorizada.

## Protocolo de la etapa 3

- El directorio `data/raw/` conserva el export original. Usar las cajas XML exclusivamente para extraer recortes de referencia. El CSV de importación registra los descartes, la partición de origen, la amplitud y el tono medio de los recortes.
- Guardar cada etiqueta humana en `data/ground_truth.csv`. Conservar `imagen` y `patente` para compatibilidad. La lista de modos de Tesseract queda en `src/ocr.py`.
- Resolver pendientes y dividir las imágenes antes del ajuste real. Mantener todas las fotos de una matrícula en una sola partición y usar únicamente `desarrollo` para ajustar. La evaluación final se reserva para una etapa posterior. Excluir el grupo si una corrección revela que cruzó la división.
- Registrar modo, parámetros, polaridad, tiempo, texto, confianza, validez y errores. La línea de base M0 conserva el RGB original. PSM 7 es la referencia y PSM 8 se ejecuta como condición separada. Conservar fallos y salidas vacías en los denominadores.
- Si faltan etiquetas verificadas de desarrollo, limitar las cifras a la demostración sintética explícitamente identificada. Mantener resultados y datos voluminosos en las rutas ignoradas por Git.

## Protocolo de la etapa 4

- El detector recibe solo la matriz de píxeles de una fotografía completa. Las cajas XML importadas se usan para visualización e IoU y quedan fuera de la selección de contornos.
- Comparar D0–D3 con los mismos filtros y puntaje geométrico. Ajustar únicamente en `desarrollo`. Mantener la evaluación final reservada. Si falta una partición de desarrollo, usar los fixtures sintéticos con ámbito explícito.
- Calcular IoU en coordenadas originales con umbrales 0,50 y 0,75. Conservar todas las imágenes en el promedio global y reportar por separado el promedio condicionado a una selección. Las fotografías anotadas permiten estimar localización de casos positivos.
- Conservar el recorte rectangular y rectificar solo con un cuadrilátero plausible. Evaluar la localización antes de la rectificación. La comparación de OCR usa el mismo preprocesamiento y PSM en ambas ramas y penaliza la ausencia de recorte automático.
- Guardar parámetros, métodos, ámbito y conjunto de IDs con los CSV. Leer los resultados desde el notebook y permitir su regeneración mediante `scripts/run_detection.py`.

## Protocolo de la etapa 4B

- Usar solo fotos, recortes y XML de desarrollo. Mantener intacta la división y reservar las 25 fotos de evaluación final.
- Preservar D0–D3, M0–M9 y los resultados históricos. Exportar los ensayos y sus parámetros a `results/stage4b/` con `scripts/run_stage4b.py`.
- Diagnosticar por separado la generación de cajas, su filtrado, su ordenamiento y el OCR. Mantener toda comparación con las etiquetas fuera del detector. Elegir configuraciones generales por accuracy y CER y conservar formas literales y confianza.
- Mantener B3_FILTRO y las dos referencias de OCR de desarrollo. El rectángulo rotado con M2/PSM 7 obtuvo el mejor accuracy automático de las variantes probadas. La banda por gradiente con inversión central y PSM 6 obtuvo el menor CER del pipeline automático combinado. Estas conclusiones describen desarrollo.

## Protocolo de la etapa 5

- Usar los 58 recortes XML de desarrollo para comparar S0–S2 y aplicar la misma configuración provisional a los recortes automáticos del detector B3_FILTRO congelado. Mantener la geometría rotada y su retorno rectangular de 4B.
- Segmentar solo con píxeles. La transcripción humana sirve para evaluar cantidad y guiar la anotación manual, que se guarda en `data/character_annotations.csv`. Conservar la división y los recortes de referencia.
- Tratar S0 con banda `fila` como provisional por conteo. Medir IoU individual y cobertura espacial sobre cajas humanas completas. La muestra de 20 patentes sigue pendiente de esas cajas.
- Preparar caracteres blancos sobre fondo negro, normalizarlos con la misma función 28 × 28 y aplanar en orden de filas a 784 valores `float32`. Mantener el alfabeto `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`. Los PNG exportados llevan índice de región y procedencia para evaluación.
