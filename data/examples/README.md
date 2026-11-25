# Recortes sintéticos de demostración

Dos imágenes creadas con `cv2.putText` y la fuente incorporada `FONT_HERSHEY_SIMPLEX` de OpenCV. Se dibujaron caracteres oscuros centrados sobre un fondo claro uniforme, con márgenes. Sirven como pruebas funcionales del OCR y cubren ambos formatos. Su geometría y sus glifos son ilustrativos.

| Archivo | Texto dibujado | Formato |
| --- | --- | --- |
| `demo_antigua_ABC123.png` | `ABC123` | Antiguo |
| `demo_mercosur_AB123CD.png` | `AB123CD` | Mercosur |

Las etiquetas de esta demostración también se indican en el notebook. `data/ground_truth.csv` queda reservado para imágenes reales. Fuente incorporada en [OpenCV](https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html) y [licencia del proyecto](https://github.com/opencv/opencv/blob/5.x/LICENSE).
