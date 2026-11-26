# OCR de patentes argentinas

Trabajo práctico final de Procesamiento Digital de Imágenes. La primera entrega aplica y evalúa técnicas clásicas para localizar patentes antiguas (`ABC123`) y Mercosur (`AB123CD`) en fotografías. El recorrido automático entrega una lectura PyTesseract y prepara regiones de caracteres de 28 × 28 para una futura red neuronal. [Resumen técnico](PREENTREGA.md) · [Guía para la defensa](DEFENSA.md) · [Notebook académico](notebooks/TP_Final_OCR.ipynb)

## Ejecutar desde CMD en Windows

Abrí CMD en la raíz del proyecto. El entorno local verificado usa Python 3.11.9 y Tesseract 5.5.3 con el idioma `eng`. Si conservás `.venv311`, podés ejecutar directamente:

```cmd
.venv311\Scripts\python.exe main.py --demo
.venv311\Scripts\python.exe main.py --entrada "C:\ruta\a\fotos" --salida "results\mis_fotos"
.venv311\Scripts\python.exe main.py --entrada "C:\ruta\a\vehiculo.jpg" --perfil cer --salida "results\una_foto"
```

`--demo` procesa ocho fotos de desarrollo documentadas en el manifiesto y escribe en `results/demo/`. Es una selección ilustrativa que incluye aciertos y fallos. `--entrada` acepta una fotografía o una carpeta. Las carpetas se recorren en orden por nombre, sin descender a subcarpetas. Usá un destino nuevo para mantener separadas las corridas y consultá el CSV de esa ejecución para encontrar sus archivos. [Manifiesto de la defensa](data/examples/preentrega.json)

| Perfil | Geometría | Preparación OCR | PSM |
| --- | --- | --- | ---: |
| `exactitud` (predeterminado) | Rotada con retorno rectangular | M2, recorte completo, polaridad original | 7 |
| `cer` | Rotada con retorno rectangular | M2, banda por gradiente, inversión según zona central | 6 |

Ambos perfiles usan B3_FILTRO, resolución nativa durante la preparación OCR, sin margen extra, idioma `eng` y lista alfanumérica ASCII. En paralelo ejecutan S0 con banda por gradiente sobre el recorte automático, con independencia de la lectura. Los parámetros se cargan de un único archivo versionado y quedan registrados en cada exportación. [Configuración del pipeline](config/pipeline_clasico.json)

### Entorno y motor OCR

Si empezás en otra máquina, instalá Python 3.11 de 64 bits con el lanzador `py` y creá el entorno. Las dependencias son NumPy, OpenCV, Matplotlib, Pandas, Pillow, PyTesseract, Jupyter Notebook y Pytest. [Python para Windows](https://www.python.org/downloads/windows/)

```cmd
py -3.11 -m venv .venv311
.venv311\Scripts\python.exe -m pip install -r requirements.txt
.venv311\Scripts\python.exe -m pip check
```

Instalá Tesseract OCR con `eng` y agregá su carpeta a `PATH`, o indicá el ejecutable con `TESSERACT_CMD`. PyTesseract se instala con pip y requiere ese motor externo para leer caracteres. En el entorno de este proyecto el pipeline y el notebook detectan la copia portátil incorporada. Los scripts históricos y las pruebas con el motor pueden apuntar explícitamente a ella. [Instalación de Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation)

```cmd
set "TESSERACT_CMD=%CD%\.venv311\Tesseract-OCR\tesseract.exe"
"%TESSERACT_CMD%" --version
"%TESSERACT_CMD%" --list-langs
.venv311\Scripts\python.exe -m pytest tests -q -rs
.venv311\Scripts\python.exe -m notebook notebooks\TP_Final_OCR.ipynb
```

Si instalaste el motor por separado, ajustá `TESSERACT_CMD` o ejecutá `set "TESSERACT_CMD="` para usar `PATH`. El pipeline conserva las ramas de detección, segmentación y normalización cuando el OCR no está disponible. En el notebook, seleccioná el kernel del entorno y usá **Kernel → Restart Kernel and Run All Cells**. Las comparaciones costosas se leen de los resultados guardados.

## Arquitectura y salidas

| Ruta | Función |
| --- | --- |
| `main.py`, `src/pipeline.py`, `src/pipeline_visualization.py` | Ejecución individual o por carpeta, estados, figuras y exportación |
| `src/preprocessing.py`, `src/detection.py` | Transformaciones M0–M9, bordes, morfología, contornos, selección y extracción |
| `src/ocr_preparation.py`, `src/ocr.py` | Acondicionamiento e invocación de Tesseract, texto y validación de formato |
| `src/segmentation.py`, `src/character_annotations.py` | Segmentación S0–S2, normalización y evaluación de cajas humanas |
| `config/`, `scripts/`, `tests/` | Perfiles congelados, experimentos reproducibles y verificación |
| `data/`, `references/`, `results/` | Dataset local, material del profesor y salidas por etapa |

Cada corrida escribe `resultados.csv`, `resultados.txt` y `config.json`. Las carpetas `detected`, `cropped`, `processed`, `characters`, `normalized` y `visualizations` guardan archivos de etapas disponibles. El CSV conserva el texto literal y normalizado, los estados, la caja semiabierta en píxeles de la foto, el perfil, la cantidad de regiones y los tiempos en milisegundos. El tiempo incluye carga y procesamiento hasta la normalización, previo a exportar figuras.

El TXT resume lo ocurrido por archivo. La confianza es la media reportada de las filas de palabra utilizables del TSV, y queda vacía si no se dispone de valores. La validez expresa si el texto cumple un formato argentino. Las regiones son candidatas sin clase. El pipeline funciona con fotos y configuración, sin consultar XML ni etiquetas humanas.

## Dataset y reproducción de experimentos

El conjunto de Roboflow Master v1 aportó 125 fotografías con cajas Pascal VOC. Se transcribieron 83 matrículas, con 58 fotos de desarrollo y 25 reservadas, y se excluyeron 23 ilegibles y 19 fuera de alcance. En una copia nueva, los experimentos reales y la demostración requieren los datos locales. El pipeline sobre fotografías nuevas es independiente de esos datos. [Fuente Roboflow Master v1](https://universe.roboflow.com/xenon-bo5e9/patentes-argentinas-master/dataset/1)

Para preparar datos en una copia nueva, descargá el export en Pascal VOC, descomprimilo en `data/raw/roboflow_argentina/` y usá `scripts.prepare_dataset --entrada data/raw/roboflow_argentina --limite 125 --semilla 42`. El importador conserva las cajas Roboflow semiabiertas, los recortes de referencia y la auditoría. `scripts.label_dataset` permite ingresar o corregir transcripciones humanas. `scripts.label_dataset --dividir --semilla 42` está destinado a una división nueva y agrupa fotos de una misma matrícula. La división de esta entrega ya está fijada.

Los siguientes comandos reproducen las etapas en subcarpetas separadas. Configurá antes el motor. Para regenerar los resultados de 4B, el script consulta también los históricos de las etapas 3 y 4 en sus ubicaciones originales. La etapa 4B completa incluye varias corridas de OCR y puede tardar.

```cmd
.venv311\Scripts\python.exe -m scripts.run_preprocessing --psm 7 --salida "results\preentrega\repro_etapa3"
.venv311\Scripts\python.exe -m scripts.run_detection --config results\detection_config.json --metodos D0 D1 D2 D3 --salida "results\preentrega\repro_etapa4"
.venv311\Scripts\python.exe -m scripts.run_stage4b --salida "results\preentrega\repro_etapa4b"
.venv311\Scripts\python.exe -m scripts.run_segmentation --config results\stage5\config.json --salida "results\stage5\repro_preentrega"
.venv311\Scripts\python.exe -m scripts.consolidar_preentrega
```

La consolidación reúne los CSV existentes de desarrollo y, si está disponible, la demostración. Genera tablas, `graphs/comparacion.png`, `examples/`, el avance de anotaciones y `limitaciones.md` en `results/preentrega/`. Los ensayos originales y las pruebas sintéticas siguen disponibles en sus carpetas históricas.

## Resultados y próxima entrega

- **B3_FILTRO** alcanzó IoU ≥ 0,50 en **26/58** y un IoU medio de **0,3618**.
- En **XML**, M2 con banda, polaridad central y PSM 6 obtuvo **5/58 exactas**, CER **0,6531**. En automático, el perfil `exactitud` obtuvo **4/58**, CER **0,7913**. El perfil `cer` obtuvo **3/58**, CER **0,7236**.
- **S0/fila** obtuvo el conteo esperado en **26/58** recortes XML y **16/58** automáticos. **S1/marco** obtuvo **17/58** en automático. Se exportaron 430 regiones sin clase individual verificada. La cantidad esperada no mide integridad espacial.
- El avance de las 20 patentes y los resultados espaciales sobre los recortes XML completos se actualizan al consolidar. Usá `.venv311\Scripts\python.exe -m scripts.label_characters --lista` o abrí el anotador omitiendo `--lista`. [Avance de la muestra](results/preentrega/anotaciones.json) · [Medición espacial](results/preentrega/espacial_xml_resumen.csv)

Son cifras de desarrollo. En el notebook se muestran errores de localización, lecturas incorrectas o vacías y regiones fusionadas, cortadas o fragmentadas. La segunda entrega queda por comenzar y abarca un dataset de 36 clases, una BPN de NumPy **784 → 64 → 36**, entrenamiento, inferencia y evaluación comparativa. [Detalles e interpretación](PREENTREGA.md)
