"""Operaciones espaciales sobre imágenes RGB o grises de 8 bits."""

import cv2
import numpy as np


def escala_grises(imagen: np.ndarray) -> np.ndarray:
    """Convierte RGB de Pillow a luminancia de 8 bits o copia la entrada gris."""
    if imagen.dtype != np.uint8 or not imagen.size:
        raise ValueError("Se espera una imagen no vacía de tipo uint8.")
    if imagen.ndim == 2:
        return imagen.copy()
    if imagen.ndim == 3 and imagen.shape[2] == 3:
        return cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    raise ValueError("Se espera una imagen RGB o una matriz de escala de grises.")


def gaussiano(gris, tamano=3, sigma=0):
    """Suaviza variaciones locales mediante un promedio con pesos gaussianos."""
    return cv2.GaussianBlur(gris, (tamano, tamano), sigma)


def mediana(gris, tamano=3):
    """Suprime ruido impulsivo usando la mediana de cada vecindad."""
    return cv2.medianBlur(gris, tamano)


def bilateral(gris, diametro=5, sigma_color=35, sigma_espacio=35):
    """Reduce el ruido ponderando distancias espaciales y de intensidad."""
    return cv2.bilateralFilter(gris, diametro, sigma_color, sigma_espacio)


def ecualizar(gris):
    """Redistribuye las intensidades según el histograma acumulado global."""
    return cv2.equalizeHist(gris)


def clahe(gris, limite=2.0, grilla=(8, 8)):
    """Ecualiza por regiones limitando la amplificación local del contraste."""
    return cv2.createCLAHE(clipLimit=limite, tileGridSize=tuple(grilla)).apply(gris)


def laplaciano(gris, tamano=3, intensidad=0.25):
    """Resta el Laplaciano con signo para realzar transiciones de intensidad."""
    derivada = cv2.Laplacian(gris, cv2.CV_32F, ksize=tamano)
    return np.clip(gris - intensidad * derivada, 0, 255).astype(np.uint8)


def high_boost(gris, tamano=3, sigma=0, intensidad=1.5):
    """Refuerza el detalle sumando un múltiplo de la máscara de desenfoque."""
    suave = gaussiano(gris, tamano, sigma)
    return cv2.addWeighted(gris, 1 + intensidad, suave, -intensidad, 0)


def umbral_global(gris, nivel=128):
    """Separa tonos claros y oscuros mediante un umbral fijo."""
    return cv2.threshold(gris, nivel, 255, cv2.THRESH_BINARY)[1]


def otsu(gris):
    """Elige el umbral global que maximiza la varianza entre dos clases."""
    return cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def umbral_adaptativo(gris, bloque=15, constante=5):
    """Usa un promedio gaussiano local para compensar iluminación irregular."""
    return cv2.adaptiveThreshold(
        gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bloque, constante
    )


def _elemento(tamano):
    return cv2.getStructuringElement(cv2.MORPH_RECT, (tamano, tamano))


def erosion(binaria, tamano=3, iteraciones=1):
    """Reduce las regiones de primer plano blanco."""
    return cv2.erode(binaria, _elemento(tamano), iterations=iteraciones)


def dilatacion(binaria, tamano=3, iteraciones=1):
    """Expande las regiones de primer plano blanco."""
    return cv2.dilate(binaria, _elemento(tamano), iterations=iteraciones)


def apertura(binaria, tamano=3, iteraciones=1):
    """Elimina detalles blancos pequeños con erosión seguida de dilatación."""
    return cv2.morphologyEx(binaria, cv2.MORPH_OPEN, _elemento(tamano), iterations=iteraciones)


def cierre(binaria, tamano=3, iteraciones=1):
    """Rellena huecos oscuros pequeños con dilatación seguida de erosión."""
    return cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, _elemento(tamano), iterations=iteraciones)


def invertir(imagen):
    """Intercambia intensidades mediante 255 menos cada píxel."""
    return cv2.bitwise_not(imagen)


def polaridad_ocr(gris):
    """Estima el fondo por mayoría en el perímetro de una máscara de Otsu."""
    mascara = otsu(gris)
    borde = np.concatenate([mascara[0, :], mascara[-1, :], mascara[1:-1, 0], mascara[1:-1, -1]])
    invertida = bool(np.any(mascara) and np.count_nonzero(borde) < borde.size / 2)
    return (invertir(gris) if invertida else gris.copy()), invertida


def limpiar_binaria_ocr(binaria, tamano=3, iteraciones=1):
    """Abre y cierra el texto blanco y devuelve texto oscuro para el OCR."""
    texto_blanco = invertir(binaria)
    limpio = cierre(apertura(texto_blanco, tamano, iteraciones), tamano, iteraciones)
    return invertir(limpio)


METODOS = {
    "M0": (),
    "M1": ((escala_grises, {}),),
    "M2": ((escala_grises, {}), (gaussiano, {"tamano": 3, "sigma": 0})),
    "M3": ((escala_grises, {}), (mediana, {"tamano": 3})),
    "M4": ((escala_grises, {}), (ecualizar, {})),
    "M5": ((escala_grises, {}), (clahe, {"limite": 2.0, "grilla": (8, 8)})),
    "M6": ((escala_grises, {}), (gaussiano, {"tamano": 3, "sigma": 0}),
           (otsu, {}), (polaridad_ocr, {})),
    "M7": ((escala_grises, {}), (clahe, {"limite": 2.0, "grilla": (8, 8)}),
           (polaridad_ocr, {}), (umbral_adaptativo, {"bloque": 15, "constante": 5})),
    "M8": ((escala_grises, {}), (high_boost, {"tamano": 3, "sigma": 0, "intensidad": 1.5}),
           (otsu, {}), (polaridad_ocr, {})),
    "M9": ((escala_grises, {}), (otsu, {}), (polaridad_ocr, {}),
           (limpiar_binaria_ocr, {"tamano": 3, "iteraciones": 1})),
}


def parametros_metodo(metodo):
    """Devuelve las operaciones en orden y sus parámetros para el registro."""
    return [{"operacion": funcion.__name__, **parametros} for funcion, parametros in METODOS[metodo]]


def aplicar_metodo(imagen: np.ndarray, metodo: str) -> tuple[np.ndarray, bool]:
    """Ejecuta un método y devuelve la imagen y la decisión de invertir."""
    salida = imagen.copy() if metodo == "M0" else imagen
    invertida = False
    for funcion, parametros in METODOS[metodo]:
        if funcion is polaridad_ocr:
            salida, invertida = funcion(salida, **parametros)
        else:
            salida = funcion(salida, **parametros)
    return salida, invertida
