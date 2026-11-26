# Primera entrega · OCR de patentes argentinas

El TP aplica procesamiento clásico a patentes antiguas y Mercosur. El recorrido automático procesa una o varias fotografías, exporta la lectura de Tesseract y prepara regiones individuales para una futura BPN. Los ensayos sobre recortes XML aíslan reconocimiento y segmentación de los fallos de localización. [Notebook académico](notebooks/TP_Final_OCR.ipynb)

## Funcionalidades y decisiones

- **Preparación espacial.** M0–M9 comparan RGB original, gris, filtros, ecualización, realce, umbralización y morfología. Se analizan imágenes intermedias y ensayos reproducibles sobre recortes de referencia.
- **Localización.** B3_FILTRO emplea gris, CLAHE, Sobel X, Otsu y cierre rectangular 19 × 5 en una imagen de lado mayor hasta 1.000 píxeles. Ajusta los filtros geométricos de D3 y conserva el puntaje. El recorte rotado admite un retorno rectangular.
- **Reconocimiento.** El perfil predeterminado `exactitud` usa M2, polaridad original y PSM 7. `cer` conserva M2 y utiliza banda por gradiente, inversión central y PSM 6. Ambos usan `eng`, la misma lista alfanumérica y escala original. El CSV guarda la salida literal antes de normalizar.
- **Segmentación.** S0 usa Otsu y contornos, S1 usa adaptativo y componentes conexas, y S2 proyecta los píxeles por columna. El pipeline aplica S0 con banda por gradiente. La normalización conserva la proporción, ajusta la dimensión mayor a 20 píxeles, centra por momentos en 28 × 28 y produce 784 valores `float32` en [0, 1].
- **Integración.** Las salidas de detección, OCR y segmentación tienen estados independientes. Se puede continuar sin el motor OCR. También se añadió `--salida` a la etapa 4B para reproducir los ensayos fuera de la carpeta histórica. La traza del detector expone la magnitud que ya calculaba, con la misma máscara y selección. [Pipeline](src/pipeline.py) · [Perfiles exactos](config/pipeline_clasico.json)

## Resultados comprobados

Roboflow Master v1 aportó 125 fotos y recortes XML. Hay 83 matrículas transcritas, 58 de desarrollo y 25 reservadas. Se excluyeron 23 ilegibles y 19 fuera de alcance. Las cifras de la tabla corresponden a las 58 fotos de desarrollo.

| Ensayo | Configuración | Resultado |
| --- | --- | --- |
| Localización | B3_FILTRO | 26/58 con IoU ≥ 0,50 · 17/58 con IoU ≥ 0,75 · IoU medio 0,3618 |
| OCR en XML | M2, banda y polaridad central · PSM 6 | 5/58 exactas · CER 0,6531 · 24 vacías |
| OCR automático · mayor acierto | B3_FILTRO, rotado, M2 · PSM 7 | 4/58 exactas · CER 0,7913 · 36 vacías |
| OCR automático · menor CER | B3_FILTRO, rotado, M2 con banda y polaridad central · PSM 6 | 3/58 exactas · CER 0,7236 · 27 vacías |
| Conteo de regiones XML / automático | S0 con banda por gradiente | 26/58 XML · 16/58 automático |
| Conteo de regiones automático | S1 con recorte de marco | 17/58 automático |

El CER micro divide las ediciones de Levenshtein acumuladas por los caracteres de referencia. Las lecturas vacías cuentan como desaciertos y eliminaciones. En el pipeline histórico D3 con PSM 7 se obtuvieron 3/58 exactas y CER 0,8726. La elección por acierto y la elección por CER corresponden a dos perfiles distintos. [Métricas y tiempos](results/preentrega/metricas.csv) · [Gráfico consolidado](results/preentrega/graphs/comparacion.png)

## Límites y pendientes

- En 25/58 fotos B3_FILTRO carecía de candidatos con IoU ≥ 0,50. En siete, había una caja admisible pero se eligió otra. Algunas cajas bien localizadas dejaban trazos fuera del recorte o daban lecturas parciales y vacías.
- Se observaron fusiones por marcos, fragmentos y bordes residuales que afectan el conteo. La cantidad esperada puede coincidir aun con caracteres cortados. Se exportaron 430 regiones sin clase verificada.
- La evaluación espacial usa las cajas humanas de los recortes XML completos, con emparejamiento uno a uno e IoU ≥ 0,50. Informamos precisión, recall, F1, IoU entre pares y proporción de patentes completas sobre ese subconjunto. El protocolo mantiene separados estos valores y el conteo sobre 58 fotos. [Avance de las 20 anotaciones](results/preentrega/anotaciones.json) · [Resultados espaciales](results/preentrega/espacial_xml_resumen.csv)
- Las ocho fotos de la demostración se eligieron para explicar transformaciones y fallos. La evaluación independiente sigue reservada para después. [Selección ilustrativa](data/examples/preentrega.json) · [Limitaciones consolidadas](results/preentrega/limitaciones.md)

## Demostración y segunda entrega

```cmd
.venv311\Scripts\python.exe main.py --demo
.venv311\Scripts\python.exe -m scripts.consolidar_preentrega
.venv311\Scripts\python.exe -m notebook notebooks\TP_Final_OCR.ipynb
```

Aprobaron 130 pruebas con Tesseract 5.5.3 y Python 3.11.9. Con el motor ausente aprobaron 124 y se omitieron seis integraciones. El notebook se ejecutó desde un kernel nuevo. Los dos perfiles reprodujeron, en las 58 fotos de desarrollo, las cajas detectadas, los píxeles que recibió el OCR y el texto normalizado guardado en 4B. El README contiene la reproducción de los experimentos. [Guía y pruebas](README.md) · [Preparación de la defensa](DEFENSA.md)

La segunda entrega propone un dataset sintético de 36 clases, una BPN **784 → 64 → 36** en NumPy, entrenamiento manual por retropropagación, inferencia y comparación con PyTesseract. Se reutilizará la normalización y se medirán los caracteres reales sobre referencias humanas. El desarrollo neuronal queda pendiente.
