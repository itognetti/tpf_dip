# OCR de patentes argentinas

Trabajo práctico final de Procesamiento Digital de Imágenes. Las etapas 1 a 4 abarcan el OCR básico, la importación de recortes de referencia, el preprocesamiento espacial y la detección clásica. La segmentación individual y la BPN corresponden a etapas posteriores.

## Configuración en Windows

Abrí PowerShell en la carpeta del proyecto. Si ya tenés `.venv311`, usá ese entorno. Los comandos de instalación se aplican a una máquina nueva con Python 3.11 de 64 bits y el lanzador `py`. [Descargas de Python para Windows](https://www.python.org/downloads/windows/)

```powershell
py -3.11 --version
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
.\.venv311\Scripts\python.exe -m pip check
```

### Motor Tesseract

`pip` instala **PyTesseract**, la biblioteca de Python. El reconocimiento requiere también **Tesseract OCR**, el motor que ejecuta la biblioteca, y los datos del idioma `eng`.

1. En una instalación nueva, descargá el instalador de 64 bits recomendado para Windows, instalá el motor y conservá los datos de inglés. [Instalación de Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation) · [Compilaciones de UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. Agregá a `Path` de Windows la carpeta que contiene `tesseract.exe` y abrí otra terminal. Verificá la versión y que `eng` aparezca en la lista.

```powershell
tesseract --version
tesseract --list-langs
```

Si preferís indicar el ejecutable para la sesión actual, usá `TESSERACT_CMD`. Ajustá la ruta si elegiste otro destino.

```powershell
$env:TESSERACT_CMD = 'C:\Program Files\Tesseract-OCR\tesseract.exe'
& $env:TESSERACT_CMD --version
& $env:TESSERACT_CMD --list-langs
```

En la carpeta de trabajo de esta entrega quedó una copia portátil del motor con `eng`, dentro del entorno virtual local. Para usarla, ejecutá lo siguiente desde la raíz en la misma sesión de PowerShell que usará Pytest o Jupyter.

```powershell
$env:TESSERACT_CMD = (Resolve-Path '.venv311/Tesseract-OCR/tesseract.exe').Path
& $env:TESSERACT_CMD --version
& $env:TESSERACT_CMD --list-langs
```

### Ejecutar el notebook y las pruebas

```powershell
.\.venv311\Scripts\python.exe -m pytest tests -q -rs
.\.venv311\Scripts\python.exe -m notebook notebooks/TP_Final_OCR.ipynb
```

En el navegador, seleccioná el kernel Python 3 del entorno virtual y elegí **Kernel → Restart Kernel and Run All Cells**. El notebook muestra la versión del motor, el dígito de referencia, los recortes y los dos resultados de OCR por imagen. Si no encuentra Tesseract o los datos `eng`, muestra los recortes y el problema de configuración. Pytest omite las cuatro pruebas de integración cuando el ejecutable o el idioma no están disponibles.

## Dataset real y etiquetado

El export local Master v1 contiene 656 JPEG y 656 XML Pascal VOC. El README del export declara dominio público, ninguna operación de preprocesamiento listada y ausencia de aumentación. Las particiones originales contienen 445 imágenes en `train`, 163 en `valid` y 48 en `test`. El dataset alternativo publica autoorientación y redimensionamiento a 512 × 512. [Master v1](https://universe.roboflow.com/xenon-bo5e9/patentes-argentinas-master/dataset/1) · [Fuente alternativa](https://universe.roboflow.com/patentes-8aydr/patentes-argentinas-ma7n9/dataset/1)

Para preparar otra copia del proyecto, descargá manualmente el export Pascal VOC y descomprimilo en `data/raw/roboflow_argentina/`. El export de esta entrega ya estaba disponible. Ejecutá desde la raíz:

```powershell
.\.venv311\Scripts\python.exe -m scripts.prepare_dataset --entrada data/raw/roboflow_argentina --limite 125 --semilla 42
```

El importador empareja archivos de igual nombre base en cada carpeta y verifica `filename`, dimensiones y una única caja `patente`. Usa coordenadas cero-basadas y límites derecho e inferior exclusivos para este export. La opción `--coordenadas voc` convierte cajas de origen uno e inclusivas. Descarta cajas fuera de rango o degeneradas y detecta copias exactas por huella de píxeles RGB decodificados entre las imágenes elegibles. Muestrea por intervalos de ancho y luminancia media con semilla fija. La partición de Roboflow se registra como procedencia, independiente de la división experimental. En una reimportación se conservan las filas y los recortes anteriores.

Se importaron **125 fotos únicas** (81 de `train`, 35 de `valid`, 9 de `test`). El inventario registra **123 descartes** (105 sin exactamente una caja, 18 por coordenadas) y **408 elegibles fuera de la muestra**. Las fotos quedaron en `data/original/` y los recortes PNG en `data/cropped/`. La auditoría se guarda en `results/dataset_import.csv`. Se verificaron los hashes de los 1.314 archivos del export y las copias seleccionadas.

`data/ground_truth.csv` conserva `imagen` (ruta del recorte relativa a `data/`) y `patente`. Agrega ID, ruta de la foto, formato, procedencia, archivo fuente, particiones original y experimental, estado, observaciones y coordenadas. Hay **83 etiquetas verificadas** (58 en desarrollo y 25 en evaluación final), 23 recortes ilegibles y 19 fuera de alcance. Para reproducir el procedimiento de etiquetado en otro conjunto:

```powershell
.\.venv311\Scripts\python.exe -m scripts.label_dataset
.\.venv311\Scripts\python.exe -m scripts.label_dataset --corregir ID_DEL_RECORTE
.\.venv311\Scripts\python.exe -m scripts.label_dataset --dividir --semilla 42
```

Cerrá la ventana de la fotografía y su recorte antes de tipear la matrícula. Podés ingresar `:ilegible`, `:fuera`, `:saltar` (continúa pendiente) o `:salir`. Cada transcripción o exclusión se guarda de inmediato. Ejecutá `--dividir` después de resolver las pendientes. La división busca aproximarse al 70/30 por número de fotos y mantiene juntas las filas de una matrícula. Si una corrección descubre una matrícula compartida entre particiones, el grupo completo queda excluido. La evaluación final queda reservada. Las transcripciones se ingresan por revisión humana.

## OCR y preprocesamiento espacial

El módulo de OCR decodifica el recorte en RGB, compone las transparencias sobre blanco y compara una línea de texto (PSM 7) con una palabra (PSM 8). Usa `eng` y limita los caracteres del motor a `A–Z` y `0–9`. Conserva el texto literal y valida la versión normalizada con una expresión regular de coincidencia completa para los formatos antiguo y Mercosur. Para la confianza consulta por separado el TSV del motor con los mismos parámetros y calcula la media aritmética de las filas de palabra no vacías con valor entre 0 y 100. Excluye el valor `-1` y devuelve `None` si no hay valores utilizables. [Módulo de OCR](src/ocr.py) · [Sintaxis y formatos de salida de Tesseract](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html)

La comparación espacial aplica M0 (original), M1 (gris), M2 (gaussiano), M3 (mediana), M4 (ecualización global), M5 (CLAHE), M6 (gaussiano y Otsu), M7 (CLAHE y adaptativo), M8 (High Boost y Otsu) y M9 (Otsu, apertura y cierre). M6–M9 estiman el fondo por el perímetro para favorecer trazos oscuros sobre fondo claro y registran si se invirtió el recorte. El notebook documenta el orden y los parámetros, y visualiza también bilateral, Laplaciano, umbral fijo, erosión, dilatación e inversión. [Implementación espacial](src/preprocessing.py)

Tras configurar `TESSERACT_CMD`, ejecutá:

```powershell
.\.venv311\Scripts\python.exe -m scripts.run_preprocessing --psm 7
.\.venv311\Scripts\python.exe -m scripts.run_preprocessing --psm 8 --salida results/psm8
```

El segundo comando ensaya el modo alternativo y guarda sus archivos en un directorio aparte. El primero escribe `results/preprocessing_experiments.csv`, `results/preprocessing_summary.csv` y tres gráficos en `results/graphs/`. Abrir y ejecutar el notebook también reproduce la comparación principal. Si faltan etiquetas verificadas de desarrollo, se usan los dos ejemplos sintéticos con un ámbito explícito en el CSV. El resumen informa accuracy y CER micro para el total y por formato. Las salidas vacías y los errores cuentan como desaciertos y como eliminaciones de los caracteres esperados. Los tiempos excluyen la lectura de disco. La confianza y la validez de formato se informan como medidas independientes del acierto.

## Material revisado

- La guía del código C describe una red 100 → 16 → 10 con entradas binarias de 10 × 10 y actualizaciones por patrón. [Guía del código C](references/bpn-c-code-student-guide.pdf)
- El tutorial deriva la sigmoide, los deltas y la actualización de pesos por descenso de gradiente a partir del error cuadrático. [Tutorial de backpropagation](references/bpn-tutorial-v16.pdf)
- El script de MNIST implementa 784 → 64 → 10 con NumPy, sigmoide, mini-batches, etiquetas one-hot, argmax y archivos NPZ. Keras se usa como cargador de MNIST. El flujo de imágenes externas usa gris, desenfoque gaussiano, Otsu invertido, dilatación, contorno de mayor área y centrado por momentos en una grilla de 28 × 28. [Script de MNIST](references/bpn_mnist_V5_trained.py)
- La imagen de 103 × 114 píxeles muestra un 2 manuscrito que sirve de entrada externa para ese flujo. [dig2.png](references/dig2.png)

La BPN prevista usará 784 → 64 → 36 y clases `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`. Conservará las operaciones vectorizadas, el cálculo manual de los gradientes y el entrenamiento por mini-batches. Requerirá pesos propios y usará las activaciones de salida como puntajes de clasificación.

## Prueba funcional de las etapas 1 y 2

El entorno de prueba usó Python 3.11.9, PyTesseract 0.3.13 y Tesseract 5.5.3.20260724 con `eng`. Las dos lecturas de cada ejemplo coincidieron con su etiqueta y pasaron la validación de formato.

| Ejemplo sintético | Predicción PSM 7 | Confianza PSM 7 | Predicción PSM 8 | Confianza PSM 8 |
| --- | --- | ---: | --- | ---: |
| `ABC123` | `ABC123` | 91,08 | `ABC123` | 0,00 |
| `AB123CD` | `AB123CD` | 90,59 | `AB123CD` | 0,00 |

El motor reportó confianza 0 para PSM 8 aun cuando leyó correctamente ambos textos. La muestra sintética verifica la ejecución de las dos configuraciones. La comparación sobre recortes reales queda pendiente.

Pytest aprobó 30 casos con el motor disponible, incluidas las cuatro integraciones. Con el motor ausente se aprobaron 26 casos y se omitieron las cuatro integraciones. El notebook se ejecutó desde un kernel nuevo en ambos escenarios.

## Observaciones de la etapa 3

Con Tesseract 5.5.3 y PSM 7 se realizaron 20 intentos sobre los dos recortes sintéticos limpios. Los diez métodos lograron 2/2 coincidencias, CER 0 y cero fallos. La confianza media varió entre 45,03 (M7) y 91,45 (M8) pese a obtener el mismo texto correcto. PSM 8 reprodujo los 20 aciertos con confianza 0. Los tiempos de esta muestra pequeña son descriptivos. El conjunto real sigue pendiente de evaluación.

En la prueba cualitativa de estrés, M0 y M7 conservaron la lectura `ABC123`, M3 y M8 añadieron caracteres o confundieron trazos, y M6 no devolvió texto. Es una única copia sintética con menor contraste y resolución, gradiente y ruido impulsivo. Se visualiza en el notebook fuera de las métricas principales.

La verificación final aprobó 75 pruebas con Tesseract. Al simular que el motor no estaba disponible, se aprobaron 71 y se omitieron cuatro integraciones. El notebook completo se ejecutó desde un kernel nuevo en el entorno existente.

Las cifras sintéticas describen el estado de la etapa 3 antes del etiquetado. La corrida de desarrollo utilizada en la etapa 4 se guarda en `results/stage3_desarrollo/`.

## Detección automática y comparación OCR

Desde PowerShell, en la raíz del proyecto:

```powershell
$env:TESSERACT_CMD = (Resolve-Path '.venv311/Tesseract-OCR/tesseract.exe').Path
.\.venv311\Scripts\python.exe -m scripts.run_preprocessing --psm 7 --salida results/stage3_desarrollo
.\.venv311\Scripts\python.exe -m scripts.run_detection --metodos D0 D1 D2 D3
.\.venv311\Scripts\python.exe -m notebook notebooks/TP_Final_OCR.ipynb
```

El notebook lee los CSV existentes y calcula en vivo unas visualizaciones individuales. La primera orden genera la comparación M0–M9 comparable y tarda varios minutos. Puede omitirse cuando ya existe la corrida con los mismos IDs y transcripciones de desarrollo. Sin ella el detector selecciona M0 como punto de partida. Para localizar sin OCR usá `--sin-ocr`. Los ensayos comparativos pueden guardarse en otro directorio con `--salida`. Usá `--sintetico` para los dos fixtures funcionales.

| Método | Preparación y máscara |
| --- | --- |
| D0 | Gris, gaussiano 5 × 5, Canny 55/150 de apertura 3 y contornos |
| D1 | D0 con cierre rectangular 13 × 5 antes de los contornos |
| D2 | Gris, gaussiano 5 × 5, Sobel X de tamaño 3, Otsu y cierre 19 × 5 |
| D3 | Gris, CLAHE de límite 2 y grilla 8 × 8 seguido de Sobel X de tamaño 3, Otsu y cierre 19 × 5 |

Se limita el lado mayor a 1.000 píxeles. Los filtros comunes usan lado largo ≥ 34 y corto ≥ 10 en la imagen de trabajo, aspecto 1,6–6,5, área relativa 0,00025–0,12, rectangularidad ≥ 0,33, densidad Canny interior 0,035–0,68 y puntaje ≥ 0,43. En el puntaje se combinan rectangularidad, aspecto, densidad y área con pesos 0,31/0,23/0,21/0,25. El componente de área vale `max(0, 1 − abs(log2(área_relativa / 0,012)) / 4)`. Se ajustó su peso en desarrollo al observar que muchas regiones incorrectas eran pequeñas. El área actúa como preferencia suave. [Configuración e implementación](src/detection.py)

La caja y el polígono se expresan en la resolución original. Se intenta la rectificación si la aproximación de contorno con epsilon 0,025 del perímetro produce cuatro vértices convexos, interiores a la foto y con al menos el 72 % del área del rectángulo rotado. Se rechazan lados menores de cuatro píxeles o lados opuestos cuya longitud difiera por un factor mayor que dos. Se conserva el recorte rectangular como alternativa. El IoU se calcula sobre la caja anterior a la homografía. Los umbrales 0,50 y 0,75 describen localización sobre fotos anotadas.

Se exportan `detection_experiments.csv`, `detection_summary.csv`, `detection_ocr_comparison.csv`, `detection_config.json` y el gráfico `results/graphs/detection_comparison.png`. Los ejemplos de caja y sus recortes quedan en `results/detected/` y `results/processed/`. Los CSV conservan los intentos fallidos y registran los principales candidatos y razones de descarte. `--config` acepta un objeto JSON con campos de `ConfigDetector` o el JSON exportado. Por ejemplo:

```powershell
.\.venv311\Scripts\python.exe -m scripts.run_detection --config results/detection_config.json --metodos D0 D1 D2 D3
```

La comparación OCR usa el detector con mejor tasa IoU 0,50 en desarrollo, con desempate por IoU 0,75 y después por IoU medio. Para el preprocesamiento se selecciona primero por coincidencia exacta y luego por CER. `--preprocesamiento` permite especificar otra opción para una corrida de desarrollo. Ambas ramas usan PSM 7 con el mismo preprocesamiento. Los casos sin detección cuentan como una matrícula enteramente omitida en el CER.

### Resultados de la etapa 4 en desarrollo

| Detector | IoU ≥ 0,50 | IoU ≥ 0,75 | IoU medio global | IoU medio con caja | Sin caja | Tiempo medio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 | 15/58 (25,86 %) | 5/58 (8,62 %) | 0,2230 | 0,2695 | 10 | 38,9 ms |
| D1 | 9/58 (15,52 %) | 1/58 (1,72 %) | 0,1379 | 0,1429 | 2 | 17,1 ms |
| D2 | 24/58 (41,38 %) | 17/58 (29,31 %) | 0,3576 | 0,3703 | 2 | 24,6 ms |
| D3 | 25/58 (43,10 %) | 16/58 (27,59 %) | 0,3499 | 0,3561 | 1 | 28,5 ms |

D3 se usó para el OCR por el criterio principal de IoU 0,50. D2 tuvo más aciertos a 0,75. En D3, 32 selecciones quedaron por debajo de IoU 0,50 y una fotografía quedó sin caja. En el ejemplo de caja incorrecta se seleccionó un borde del vehículo del fondo. En el caso sin caja, el contorno de mayor IoU (0,801) se descartó por área relativa y rectangularidad. Las tasas se obtuvieron en el mismo conjunto empleado para ajustar el puntaje.

La comparación M0–M9 de desarrollo eligió M2 por 4 coincidencias y CER 0,8211. M3 también acertó 4, con CER 0,8238. La comparación con Tesseract 5.5.3, M2 y PSM 7 dio:

| Entrada | Coincidencia exacta | CER micro | Sin texto | Sin recorte |
| --- | ---: | ---: | ---: | ---: |
| Recorte XML | 4/58 (6,90 %) | 0,8211 | 40 | 0 |
| Recorte automático D3 | 3/58 (5,17 %) | 0,8726 | 35 | 1 |

De 57 recortes automáticos, 3 se rectificaron y 54 conservaron el rectángulo. En el ejemplo de rectificación, el cuadrilátero contiene parte del paragolpes y la transformación deja sesgada la patente. Entre las 25 cajas de D3 con IoU ≥ 0,50 hubo 14 lecturas vacías. La evaluación final de 25 fotos queda reservada. Antes de avanzar, conviene revisar con desarrollo qué ajustes reducen la selección de regiones ajenas y los OCR vacíos en recortes de referencia.

La etapa 4 aprobó 92 pruebas con Tesseract y 88 con el motor ausente, con cuatro integraciones omitidas en ese caso. El notebook se verificó desde un kernel nuevo en el entorno existente.

Los puntos de entrada `main.py` y `train.py` quedan reservados. [Arquitectura y etapas](AGENTS.md)

## Etapa 4B · Diagnóstico y optimización en desarrollo

### Ejecutar en CMD

Desde la raíz del proyecto, con el entorno virtual existente:

```cmd
set "TESSERACT_CMD=%CD%\.venv311\Tesseract-OCR\tesseract.exe"
.venv311\Scripts\python.exe -m scripts.run_stage4b --trabajadores 4
.venv311\Scripts\python.exe -m pytest tests -q -rs
.venv311\Scripts\python.exe -m notebook notebooks\TP_Final_OCR.ipynb
```

### Ejecutar en PowerShell

```powershell
$env:TESSERACT_CMD = (Resolve-Path '.venv311/Tesseract-OCR/tesseract.exe').Path
.\.venv311\Scripts\python.exe -m scripts.run_stage4b --trabajadores 4
.\.venv311\Scripts\python.exe -m pytest tests -q -rs
.\.venv311\Scripts\python.exe -m notebook notebooks/TP_Final_OCR.ipynb
```

La orden de experimentación debe encontrar los CSV y la configuración de la etapa 4 en `results/`, y el export Pascal VOC en `data/raw/roboflow_argentina/`. Puede tardar varios minutos. `--fase ocr`, `--fase detector` y `--fase pipeline` permiten ejecutar cada parte por separado, con el pipeline después de obtener las dos etapas parciales. Las pruebas y el notebook se ejecutan desde el entorno existente. La evaluación y la selección usan los mismos 58 IDs de desarrollo. Los 25 casos de evaluación final quedan reservados. El notebook carga los resultados guardados.

### Auditoría y OCR

Se verificaron las 58 correspondencias XML, caja semiabierta y píxeles de los recortes. Se verificaron las etiquetas contra los registros históricos y se revisaron visualmente los recortes de desarrollo. La entrada histórica era M2, luminancia a partir de RGB y gaussiano 3 × 3, con matriz gris de 8 bits, idioma `eng` y whitelist `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`. Se reprodujeron los 58 textos literales XML y los 58 del pipeline automático de la etapa 4. Los PNG de depuración conservan los píxeles entregados al motor.

Los recortes abarcan 59–1277 píxeles de ancho y 35–424 de alto. La mediana de altura fue 104 píxeles en las 40 lecturas vacías y 87 en las otras 18. Hubo vacíos en ambos formatos, 25/37 antiguas y 15/21 Mercosur. En los ejemplos se ven caracteres de aproximadamente 15–20 píxeles de alto en el recorte de 65 × 35, además de inclinación, baja nitidez, franjas, marcos y tornillos. Esas condiciones siguen siendo hipótesis causales para casos concretos. El efecto de PSM sí se aisló con píxeles fijos: PSM 7, 6, 8 y 13 dieron 40, 31, 15 y 14 vacíos, respectivamente, con cuatro aciertos exactos cada uno. [Guía de Tesseract](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html)

Se ensayaron 28 configuraciones de OCR en cuatro rondas. Se compararon factores 1–4, bicúbica y lineal, altura objetivo de 96 y margen blanco de 12. Siguieron inversiones por perímetro y zona central, gris, gaussiano, CLAHE, Otsu, umbral adaptativo, High Boost, morfología, márgenes y recorte de banda por energía del gradiente. Después se compararon PSM y se combinaron finalistas. Se conserva una selección global por accuracy y CER. Las salidas TXT y TSV se obtienen en una sola invocación por ensayo mediante PyTesseract.

El OCR aislado elegido, `PSM6` en los CSV, usa banda horizontal conservadora, polaridad central y M2 a tamaño nativo. Logró 5/58 aciertos y CER 0,6531. `B0` obtuvo 4/58 y CER 0,8211. Los 5 aciertos se reparten en 4/37 patentes antiguas y 1/21 Mercosur. La lectura de `AD252JL` se recuperó. En otros casos persisten sustituciones y caracteres extra, como `AF362FFI` para `AF362EF`, o inclusión de texto de la carrocería. [Entradas OCR](results/stage4b/examples/ocr_inputs.png) · [Resumen OCR](results/stage4b/ocr_summary.csv)

### Candidatos, geometría y comparación integrada

Se midieron D0–D3 y nueve variantes. Cambian resolución, gaussiano, tamaño Sobel, cierre, dilatación, dimensiones relativas, filtros y puntaje por distribución vertical. Se conservaron las cajas históricas de D0–D3. El CSV de candidatos registra 129.789 hipótesis y el CSV de intentos conserva el número total de contornos y los descartes. Los candidatos requieren un soporte mínimo de 4 × 3 píxeles. Top 3 y top 5 se calculan entre los candidatos que pasaron los filtros, con 58 imágenes en el denominador.

| Detector | Correcta antes del filtro | Después del filtro | Top 3 | Top 5 | Selección IoU ≥ 0,50 | A / B / C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D3 | 33/58 | 30/58 | 29/58 | 29/58 | 25/58 | 25 / 3 / 5 |
| B3_FILTRO | 33/58 | 33/58 | 31/58 | 31/58 | 26/58 | 25 / 0 / 7 |

B3_FILTRO mantiene la generación de D3 y ajusta filtros generales: aspecto 1,45–7, área 0,00018–0,26, rectangularidad mínima 0,23, densidad 0,028–0,68 y lados mínimos 30 y 8. La caja histórica con IoU 0,801 pasó los filtros y quedó seleccionada, aunque deja fuera parte del último dígito. La principal limitación sigue siendo la generación de candidatos. B2_MULTI obtuvo 20/58 aciertos a IoU ≥ 0,75 y un IoU global de 0,3800, el mayor del ensayo. B3_FILTRO se eligió por la tasa principal a IoU ≥ 0,50. [Análisis de candidatos](results/stage4b/graphs/detection_candidates.png) · [Caso histórico](results/stage4b/examples/candidato_descartado.png)

La política de rectángulo rotado obtuvo 3/58 coincidencias y CER 0,7236 con el OCR elegido. La caja rectangular obtuvo 1/58 y 0,8428. La perspectiva estricta se aplicó en dos casos y obtuvo 1/58 con CER 0,8293, contando el retorno a la caja rectangular. El recorte rotado conserva la caja rectangular como alternativa cuando su geometría es inestable.

| Pipeline | IoU medio | IoU ≥ 0,50 / 0,75 | Exactas | CER | Vacías | Sin caja |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Anterior | 0,3499 | 43,10 % / 27,59 % | 3/58 | 0,8726 | 35 | 1 |
| Solo OCR optimizado | 0,3499 | 43,10 % / 27,59 % | 1/58 | 0,8347 | 30 | 1 |
| Detector y recorte optimizados | 0,3618 | 44,83 % / 29,31 % | 4/58 | 0,7913 | 36 | 0 |
| Ambas optimizaciones | 0,3618 | 44,83 % / 29,31 % | 3/58 | 0,7236 | 27 | 0 |
| XML con OCR optimizado | — | — | 5/58 | 0,6531 | 24 | — |

Las mejoras en CER y lecturas vacías del pipeline combinado no aumentaron las patentes exactas. Para accuracy automático, la referencia provisional es B3_FILTRO con recorte rotado y M2/PSM 7. Para CER, la referencia es la combinación con banda, inversión central y PSM 6. Las cifras describen el conjunto usado para seleccionar parámetros. Antes de pasar a la segmentación, conviene verificar primero la cobertura de la fila sobre XML y luego sobre cajas automáticas, con ejemplos de ambas polaridades, inclinación y caracteres cortados. [Comparación integrada](results/stage4b/pipeline_summary.csv) · [Parámetros reproducibles](results/stage4b/config.json)

La verificación aprobó 104 pruebas con Tesseract. Con el motor ausente pasaron 99 y se omitieron cinco integraciones. El notebook completo se ejecutó desde un kernel nuevo.

### Reordenamiento por evidencia de caracteres

El puntaje geométrico no distingue una patente de otras regiones rectangulares con texto, como insignias del modelo o calcomanías. En 7 de las 58 fotos de desarrollo, B3_FILTRO admitió una caja correcta y seleccionó otra. En esos casos la diferencia la decidía el área, la rectangularidad o la densidad según la foto, y las patentes correctas ocupaban entre 0,2 % y 22 % de la imagen.

B3_CARACTERES conserva la generación y los filtros de B3_FILTRO. Toma los tres candidatos admitidos de mayor puntaje, segmenta cada recorte con S0 y banda `fila` y calcula una evidencia entre 0 y 1. La evidencia vale 1 con seis o siete regiones, 0,5 con cinco u ocho y 0,2 en otro caso, multiplicada por la uniformidad de las alturas. Con menos de cuatro regiones vale 0. Se elige el máximo de `puntaje + 0,5 · evidencia`. La selección usa solo píxeles. Los parámetros están en `ConfigDetector.reordenar_k` y `ConfigDetector.peso_caracteres`, y los valores por defecto conservan la selección de D0–D3 y B3_FILTRO. [Implementación](src/detection.py) · [Evidencia](src/segmentation.py)

Con Tesseract 5.4.0, sobre las mismas 58 fotos de desarrollo:

| Detector | IoU ≥ 0,50 | IoU ≥ 0,75 | IoU medio | C · otra caja | Tiempo medio |
| --- | ---: | ---: | ---: | ---: | ---: |
| B3_FILTRO | 26/58 | 17/58 | 0,3618 | 7 | 18,7 ms |
| B3_CARACTERES | 30/58 | 18/58 | 0,4156 | 3 | 24,0 ms |

Se ganaron cuatro patentes y no se perdió ninguna. Con el OCR elegido (B0P8) y recorte rectangular, el pipeline combinado mantuvo 5/58 aciertos exactos y bajó el CER de 0,7588 a 0,7154, con 19 lecturas vacías en lugar de 21. Varias lecturas nuevas quedaron a un carácter, por ejemplo `HWO350` para HWQ350. Ampliar la tolerancia del área en lugar de reordenar perdió tres patentes. Se probaron cinco variantes en el mismo conjunto de desarrollo, por lo que la mejora debe confirmarse en la evaluación final reservada.

La etapa 5 conserva B3_FILTRO por defecto. Para segmentar los recortes del nuevo detector en un directorio aparte:

```powershell
.\.venv312\Scripts\python.exe -m scripts.run_segmentation --detector B3_CARACTERES --salida results/stage5/caracteres
```

Con S0 y banda `fila`, la cantidad correcta en recortes automáticos pasó de 15/58 a 18/58.

## Etapa 5 · Segmentación y normalización individual

Desde la raíz del proyecto en **CMD**, usá el entorno ya instalado:

```cmd
.venv311\Scripts\python.exe -m scripts.run_segmentation
.venv311\Scripts\python.exe -m scripts.label_characters --lista
.venv311\Scripts\python.exe -m scripts.label_characters
```

El experimento requiere la configuración guardada en `results/stage4b/config.json`. Procesa primero los recortes XML de desarrollo y luego usa B3_FILTRO con la geometría rotada de la etapa 4B y su retorno rectangular. Para reproducir una combinación en un directorio aparte:

```cmd
.venv311\Scripts\python.exe -m scripts.run_segmentation --tipo xml --metodos S0 S2 --bandas completa fila --config results\stage5\config.json --salida results\stage5\ensayo_xml
.venv311\Scripts\python.exe -m scripts.label_characters --id ID_DEL_RECORTE
```

Para ejecutar las integraciones de las etapas anteriores y abrir el notebook:

```cmd
set "TESSERACT_CMD=%CD%\.venv311\Tesseract-OCR\tesseract.exe"
.venv311\Scripts\python.exe -m pytest tests -q -rs
.venv311\Scripts\python.exe -m notebook notebooks\TP_Final_OCR.ipynb
```

### Métodos y normalización

Los métodos comparan región completa, recorte moderado de márgenes uniformes y banda por gradiente. Se trabaja a 96 píxeles de alto, con relación de aspecto conservada, CLAHE de límite 1,5 y polaridad estimada en el interior. Se retiran líneas horizontales largas en franjas periféricas. S0 usa Otsu y contornos externos. S1 usa umbral adaptativo gaussiano con bloque 25 y constante 7, seguido de componentes conexas de ocho vecinos. S2 usa Otsu y proyección vertical. Su umbral de 25 % de la altura se fijó en una ronda breve de desarrollo que comparó 10, 15, 20 y 25 %. Los criterios de geometría, densidad, alineación, solapamiento y reparación moderada se registran en el JSON. Cada método usa los píxeles del recorte.

`normalizar_caracter` ajusta la caja al primer plano, preserva la proporción, escala el lado mayor a 20 por defecto y centra con el centro de masa en una matriz de 28 × 28. Retorna la matriz y el vector de 784 componentes `float32` en [0, 1]. Los caracteres impresos preservan sus trazos sin aplicar una dilatación fija. El orden de clases previsto es `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`. [Referencia del profesor](references/bpn_mnist_V5_trained.py) · [Momentos de imagen en OpenCV](https://docs.opencv.org/4.x/d0/d49/tutorial_moments.html)

### Resultados en desarrollo

Cantidad correcta de regiones sobre 58 recortes por variante:

| Método | XML completa | XML marco | XML fila | Auto completa | Auto marco | Auto fila |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 | 8 | 9 | **26** | 11 | 11 | 16 |
| S1 | 14 | 14 | 15 | 12 | **17** | 11 |
| S2 | 6 | 4 | 6 | 2 | 1 | 0 |

Se eligió **S0 con banda fila** por obtener el mayor número de coincidencias de cantidad en XML. Allí hubo 4 casos con exceso y 28 con faltantes, incluidos 13 sin regiones. Con la misma configuración en automático hubo 5 excesos y 37 faltantes, incluidos 22 sin regiones. El detector produjo 58 recortes y 32 tuvieron IoU de patente inferior a 0,50. En 15 fotos coincidió la cantidad y el IoU alcanzó 0,50. Las cantidades no certifican la integridad ni la correspondencia de cada región.

En el caso de IoU 0,801, la extracción automática cubrió aproximadamente 79,8 % de la caja XML y perdió casi todo el último dígito. Se obtuvieron seis regiones en ambos recortes, incluida una franja residual en el automático. También se observaron candidatos de marco y fusiones de texto con elementos periféricos. Una unión o separación propuesta por el algoritmo requiere verificación visual. [Recorte incompleto](results/stage5/examples/automatico_recortado.png) · [Comparación visual](results/stage5/examples/xml_comparacion.png)

Los CSV de experimentación y resumen guardan cada intento, cajas en coordenadas del recorte de entrada, banda, fallos y tiempos. `characters.csv` registra **430 regiones** de la configuración provisional, 249 XML y 181 automáticas, con rutas a las máscaras y a los PNG normalizados. El índice describe la posición de izquierda a derecha. La clase se registra únicamente en las anotaciones humanas. [Resumen](results/stage5/segmentation_summary.csv) · [Muestra normalizada](results/stage5/examples/normalizacion.png)

La verificación de la etapa 5 aprobó 122 pruebas con Tesseract. Con el motor ausente pasaron 117 y se omitieron cinco integraciones. El notebook completo se ejecutó desde un kernel nuevo y se comprobaron las dimensiones de los 430 PNG normalizados.

### Anotación manual y siguiente paso

La muestra contiene 20 recortes reproducibles, diez de cada formato y tres niveles aproximados de dificultad. En la ventana de Matplotlib, dibujá con el botón izquierdo sobre el carácter que indica el título. Usá flechas para cambiar de posición, **X** para omitir, **Retroceso** para borrar, **Enter** para pasar a la siguiente foto y **Esc** para salir. Cada cambio se guarda en `data/character_annotations.csv`. `--id` abre un recorte de desarrollo para corregirlo. El comando principal retoma posiciones pendientes.

Hay cero cajas humanas. IoU por carácter, precisión, recall, F1, cobertura de banda y tasa de patentes con todos sus caracteres segmentados quedan pendientes. Al repetir el experimento con las cajas completas se exportan las métricas espaciales XML y diagnósticos de cortes, uniones y fragmentación. Las fotografías reservadas quedan fuera del ensayo.

Para preparar el futuro dataset neuronal, conviene verificar la clase y la integridad de cada recorte, conservar ambas polaridades de origen en los metadatos y aplicar esta misma normalización. Agrupá variantes derivadas de un mismo glifo base al separar los conjuntos. Los recortes reales de la etapa 5 se conservan para evaluación.
