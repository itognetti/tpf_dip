# Guía breve para la defensa

## Recorrido de 5 minutos

1. **Problema.** Localizar patentes argentinas, leerlas y preparar entradas de caracteres usando fotografías. Relacioná el recorrido con filtros espaciales, histogramas, gradientes, morfología, contornos y momentos.
2. **Demostración.** En CMD, ejecutá `.venv311\Scripts\python.exe main.py --demo`. Abrí `results/demo/resultados.txt` y dos figuras de `results/demo/visualizations/`. La primera foto muestra un acierto. La cuarta muestra una patente Mercosur bien localizada con un carácter extra. La sexta muestra un dígito cortado pese a IoU 0,801 y a seis regiones.
3. **Resultados.** B3_FILTRO localizó 26/58 fotos con IoU ≥ 0,50. El mejor acierto automático fue 4/58 con PSM 7. PSM 6 dio 3/58 y el mejor CER automático, 0,7236. S0/fila devolvió la cantidad esperada en 26/58 recortes XML y 16/58 automáticos. Las cifras describen desarrollo.
4. **Aprendizaje.** La figura permite distinguir problemas de localización, lectura y segmentación. Una buena caja o seis regiones pueden ocultar el corte de un trazo. La segunda entrega mantendrá estas referencias para comparar PyTesseract con una BPN de NumPy, una vez entrenada y evaluada.

[Notebook](notebooks/TP_Final_OCR.ipynb) · [Selección completa](data/examples/preentrega.json) · [Resultados verificados](PREENTREGA.md)

## Preguntas técnicas probables

| Pregunta | Respuesta breve |
| --- | --- |
| ¿En qué difieren Sobel y Canny? | Sobel estima derivadas. El eje X destaca bordes verticales de los caracteres. Canny afina máximos locales y usa histéresis para conservar bordes conectados a respuestas fuertes. |
| ¿Por qué un cierre rectangular 19 × 5? | Favorece consolidar zonas separadas horizontalmente en la escala de trabajo de D3. Si une la patente con otras estructuras, también puede deformar el candidato. |
| ¿Cuándo sirve Otsu o el umbral adaptativo? | Otsu elige un umbral global que maximiza la separación entre dos clases de intensidades. El adaptativo usa el contexto local y puede responder mejor a iluminación desigual, a costa de resaltar ruido. |
| ¿Cómo se elige un contorno? | Primero se filtra por aspecto, área, lados, rectangularidad y densidad de bordes. Entre candidatos elegibles se elige el mayor puntaje geométrico. |
| ¿Qué mide IoU? | El área de intersección dividida por la de unión de dos cajas. Mide localización de la patente. Permite que una región con IoU alto corte un carácter. |
| ¿En qué difieren CER y exactitud? | La exactitud exige que coincida toda la matrícula. El CER micro cuenta sustituciones, inserciones y eliminaciones por carácter de referencia. Una lectura parcial puede reducir CER sin sumar aciertos exactos. |
| ¿Qué limitó a PyTesseract? | Hubo lecturas vacías, sustituciones y texto extra incluso con recortes bien localizados. PSM 6 redujo el CER, pero produjo un acierto automático menos que PSM 7. |
| ¿Qué aporta S0 frente a S1 y S2? | S0 agrupa por contornos en la binaria. S1 usa componentes conexas con umbral local. S2 busca valles en la proyección vertical. Una línea residual puede unir regiones o llenar esos valles. |
| ¿Cómo se centra un carácter? | Los momentos ponderan coordenadas por intensidad. El desplazamiento acerca el centro de masa al centro del lienzo preservando el trazo dentro de sus límites. |
| ¿Por qué 28 × 28 y 784 componentes? | Reutiliza el esquema de entradas del ejemplo MNIST del profesor. Se ajusta el lado mayor a 20 píxeles, se centra en 28 × 28 y se aplana por filas para la BPN propuesta. |
| ¿Qué se puede atribuir a cada etapa? | Se comprobó un caso con corte de un dígito, lecturas vacías sobre XML y efectos de marcos y fragmentos en la segmentación. La concordancia se contrasta con cajas humanas en el subconjunto XML completo. El consolidado informa su denominador. |
| ¿Por qué dos entregas? | La primera deja un recorrido clásico medido y una representación reutilizable. La segunda requiere crear datos de entrenamiento, ajustar una BPN para 36 clases y comparar el reconocimiento con referencias verificadas. |
