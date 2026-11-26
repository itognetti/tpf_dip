"""Localización clásica a partir de una fotografía completa, en RGB o gris."""

from dataclasses import dataclass
from math import ceil, floor, isclose, log2

import cv2
import numpy as np

from .preprocessing import clahe, escala_grises, gaussiano, otsu

Caja = tuple[int, int, int, int]
METODOS_DETECCION = ("D0", "D1", "D2", "D3")
OPERACIONES = {
    "dilatacion": cv2.MORPH_DILATE,
    "erosion": cv2.MORPH_ERODE,
    "apertura": cv2.MORPH_OPEN,
    "cierre": cv2.MORPH_CLOSE,
}


@dataclass(frozen=True)
class ConfigDetector:
    max_lado: int = 1000
    gauss_kernel: int = 5
    canny_bajo: float = 55
    canny_alto: float = 150
    canny_kernel: int = 3
    sobel_kernel: int = 3
    clahe_limite: float = 2.0
    clahe_grilla: tuple[int, int] = (8, 8)
    morfologia: str = "cierre"
    kernel_canny: tuple[int, int] = (13, 5)
    kernel_sobel: tuple[int, int] = (19, 5)
    iteraciones: int = 1
    kernel_relativo_sobel: tuple[float, float] | None = None
    kernel_dilatacion: tuple[int, int] | None = None
    iteraciones_dilatacion: int = 1
    aspecto: tuple[float, float] = (1.6, 6.5)
    area_relativa: tuple[float, float] = (0.00025, 0.12)
    ancho_min: int = 34
    alto_min: int = 10
    rectangularidad_min: float = 0.33
    densidad: tuple[float, float] = (0.035, 0.68)
    area_objetivo: float = 0.012
    pesos: tuple[float, float, float, float] = (0.31, 0.23, 0.21, 0.25)
    peso_vertical: float = 0.0
    puntaje_min: float = 0.43
    epsilon: float = 0.025
    reordenar_k: int = 1
    peso_caracteres: float = 0.0

    def __post_init__(self):
        pares = (self.kernel_canny, self.kernel_sobel, self.clahe_grilla,
                 self.aspecto, self.area_relativa, self.densidad)
        pares += tuple(p for p in (self.kernel_relativo_sobel, self.kernel_dilatacion) if p is not None)
        if any(len(par) != 2 for par in pares):
            raise ValueError("Cada intervalo o tamaño debe tener dos componentes.")
        enteros = (self.max_lado, self.gauss_kernel, self.ancho_min, self.alto_min,
                   self.iteraciones, self.iteraciones_dilatacion, *self.kernel_canny,
                   *self.kernel_sobel, *self.clahe_grilla, *(self.kernel_dilatacion or ()))
        if any(not isinstance(valor, int) or valor < 1 for valor in enteros):
            raise ValueError("Las dimensiones, núcleos e iteraciones deben ser enteros positivos.")
        if (self.gauss_kernel % 2 != 1 or self.canny_kernel not in (3, 5, 7)
                or self.sobel_kernel not in (1, 3, 5, 7)):
            raise ValueError("Revisá el tamaño de los núcleos gaussianos, de bordes y rectangulares.")
        if (not 0 <= self.canny_bajo < self.canny_alto or not self.clahe_limite > 0
                or self.morfologia not in OPERACIONES):
            raise ValueError("Revisá los umbrales de Canny, CLAHE y la operación morfológica.")
        if (not 1 <= self.aspecto[0] < self.aspecto[1]
                or not 0 <= self.area_relativa[0] < self.area_relativa[1] <= 1
                or not 0 <= self.densidad[0] < self.densidad[1] <= 1
                or not 0 <= self.rectangularidad_min <= 1 or not 0 <= self.puntaje_min <= 1
                or not 0 < self.area_objetivo <= 1
                or len(self.pesos) != 4 or not all(0 <= p <= 1 for p in self.pesos)
                or not isclose(sum(self.pesos), 1.0)
                or not 0 <= self.peso_vertical <= 1
                or (self.kernel_relativo_sobel is not None and
                    not all(0 < valor <= 0.25 for valor in self.kernel_relativo_sobel))
                or not 0 < self.epsilon < 0.25):
            raise ValueError("Revisá los intervalos geométricos, el puntaje y epsilon.")
        if (not isinstance(self.reordenar_k, int) or self.reordenar_k < 1
                or not 0 <= self.peso_caracteres <= 2):
            raise ValueError("El reordenamiento usa k ≥ 1 y un peso de caracteres entre 0 y 2.")


@dataclass
class Candidato:
    caja: Caja
    contorno: np.ndarray
    rectangulo_rotado: np.ndarray
    poligono: np.ndarray
    superficie: float
    aspecto: float
    area_relativa: float
    rectangularidad: float
    densidad: float
    puntaje: float
    rechazos: tuple[str, ...]
    distribucion_vertical: float | None = None
    escala: int | None = None
    evidencia: float | None = None


@dataclass
class Deteccion:
    candidatos: list[Candidato]
    seleccionado: Candidato | None
    motivo: str
    imagenes: dict[str, np.ndarray]
    n_contornos: int = 0


@dataclass
class Extraccion:
    rectangular: np.ndarray
    rectificada: np.ndarray | None
    motivo: str


def preparar_gris(imagen: np.ndarray, max_lado: int) -> np.ndarray:
    """Valida la imagen y reduce su lado mayor conservando las proporciones."""
    gris = escala_grises(imagen)
    alto, ancho = gris.shape
    if max(alto, ancho) <= max_lado:
        return gris
    factor = max_lado / max(alto, ancho)
    tamano = max(1, round(ancho * factor)), max(1, round(alto * factor))
    return cv2.resize(gris, tamano, interpolation=cv2.INTER_AREA)


def canny(gris: np.ndarray, bajo=55, alto=150, kernel=3) -> np.ndarray:
    """Aplica supresión de no máximos y umbrales con histéresis."""
    return cv2.Canny(gris, bajo, alto, apertureSize=kernel, L2gradient=True)


def sobel(gris: np.ndarray, kernel=3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve Gx y Gy con signo y la magnitud euclídea en float32."""
    gx = cv2.Sobel(gris, cv2.CV_32F, 1, 0, ksize=kernel)
    gy = cv2.Sobel(gris, cv2.CV_32F, 0, 1, ksize=kernel)
    return gx, gy, cv2.magnitude(gx, gy)


def intensidad_gradiente(gradiente: np.ndarray) -> np.ndarray:
    """Escala |G| de cero a 255 sin saturar las derivadas antes de Otsu."""
    absoluta = np.abs(gradiente)
    maximo = absoluta.max()
    if maximo == 0:
        return np.zeros_like(absoluta, dtype=np.uint8)
    return np.rint(absoluta * (255.0 / maximo)).astype(np.uint8)


def morfologia_rectangular(binaria, operacion="cierre", tamano=(13, 5), iteraciones=1):
    """Aplica una operación sobre primer plano blanco, con (ancho, alto)."""
    elemento = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(tamano))
    return cv2.morphologyEx(binaria, OPERACIONES[operacion], elemento, iterations=iteraciones)


def rectangularidad(superficie: float, ancho: float, alto: float) -> float:
    """Fracción de área ocupada en el rectángulo rotado de área mínima."""
    return min(1.0, superficie / (ancho * alto)) if ancho > 0 and alto > 0 else 0.0


def caja_original(caja: Caja, forma_trabajo: tuple, forma_original: tuple) -> Caja:
    """Convierte una caja semiabierta y la limita a las dimensiones originales."""
    alto, ancho = forma_original[:2]
    sy, sx = alto / forma_trabajo[0], ancho / forma_trabajo[1]
    x0, y0, x1, y1 = caja
    return (max(0, min(ancho, floor(x0 * sx))), max(0, min(alto, floor(y0 * sy))),
            max(0, min(ancho, ceil(x1 * sx))), max(0, min(alto, ceil(y1 * sy))))


def _puntos_originales(puntos, trabajo, original):
    return puntos.reshape(-1, 2).astype(np.float32) * np.array(
        [original[1] / trabajo[1], original[0] / trabajo[0]], dtype=np.float32
    )


def extraer_candidatos(
    mascara, bordes, forma_original, config: ConfigDetector, *, contornos=None, bordes_verticales=None,
) -> list[Candidato]:
    """Describe los contornos y aplica criterios iguales a todos los métodos."""
    if contornos is None:
        contornos, _ = cv2.findContours(mascara, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    for contorno in contornos:
        x, y, ancho, alto = cv2.boundingRect(contorno)
        if ancho < 4 or alto < 3:
            continue
        caja = x, y, x + ancho, y + alto
        superficie = cv2.contourArea(contorno)
        rotado = cv2.minAreaRect(contorno)
        corto, largo = sorted(rotado[1])
        aspecto = largo / corto if corto else 0.0
        area = ancho * alto / mascara.size
        rect = rectangularidad(superficie, largo, corto)
        dx, dy = ancho // 10, alto // 10
        interior = bordes[y + dy:y + alto - dy, x + dx:x + ancho - dx]
        densidad = cv2.countNonZero(interior) / interior.size
        puntaje_aspecto = min(1.0, aspecto / 2.1, 4.1 / aspecto) if aspecto else 0.0
        puntaje_densidad = max(0.0, 1.0 - abs(densidad - 0.20) / 0.28)
        puntaje_area = max(0.0, 1.0 - abs(log2(area / config.area_objetivo)) / 4)
        componentes = rect, puntaje_aspecto, puntaje_densidad, puntaje_area
        puntaje = sum(peso * valor for peso, valor in zip(config.pesos, componentes))
        vertical = None
        if bordes_verticales is not None:
            distribucion = bordes_verticales[y + dy:y + alto - dy, x + dx:x + ancho - dx]
            vertical = distribucion_vertical(distribucion)
            puntaje = (1 - config.peso_vertical) * puntaje + config.peso_vertical * vertical
        criterios = (
            ("dimensiones", largo >= config.ancho_min and corto >= config.alto_min),
            ("aspecto", config.aspecto[0] <= aspecto <= config.aspecto[1]),
            ("area_relativa", config.area_relativa[0] <= area <= config.area_relativa[1]),
            ("rectangularidad", rect >= config.rectangularidad_min),
            ("densidad", config.densidad[0] <= densidad <= config.densidad[1]),
            ("puntaje", puntaje >= config.puntaje_min),
        )
        poligono = cv2.approxPolyDP(contorno, config.epsilon * cv2.arcLength(contorno, True), True)
        candidatos.append(Candidato(
            caja_original(caja, mascara.shape, forma_original),
            _puntos_originales(contorno, mascara.shape, forma_original),
            _puntos_originales(cv2.boxPoints(rotado), mascara.shape, forma_original),
            _puntos_originales(poligono, mascara.shape, forma_original),
            superficie, aspecto, area, rect, densidad, puntaje,
            tuple(nombre for nombre, valido in criterios if not valido),
            vertical, config.max_lado,
        ))
    return sorted(candidatos, key=lambda c: (-c.puntaje, c.caja))


def seleccionar_candidato(candidatos: list[Candidato], imagen: np.ndarray | None = None,
                          config: ConfigDetector | None = None) -> Candidato | None:
    """Escoge el de mayor puntaje entre los que pasan todos los criterios.

    Con reordenar_k > 1 y peso_caracteres > 0, segmenta los k mejores admitidos y suma la
    evidencia de caracteres ponderada al puntaje geométrico. Usa solo píxeles de la foto.
    """
    admitidos = sorted((c for c in candidatos if not c.rechazos), key=lambda c: (-c.puntaje, c.caja))
    if not admitidos:
        return None
    if imagen is None or config is None or config.reordenar_k < 2 or config.peso_caracteres == 0:
        return admitidos[0]
    from .segmentation import evidencia_caracteres

    mejores = admitidos[:config.reordenar_k]
    for candidato in mejores:
        candidato.evidencia = evidencia_caracteres(recortar_rectangulo(imagen, candidato.caja))[0]
    return max(mejores, key=lambda c: c.puntaje + config.peso_caracteres * c.evidencia)


def distribucion_vertical(bordes: np.ndarray) -> float:
    """Combina presencia y equilibrio de bordes verticales en cuatro franjas."""
    fracciones = [np.mean(z) for z in np.array_split(bordes, 4, axis=1) if z.size]
    if not fracciones or max(fracciones) == 0:
        return 0.0
    return float(min(1.0, np.mean(fracciones) / 0.12) * np.mean(fracciones) / max(fracciones))


def kernel_relativo(forma, factores):
    """Convierte fracciones de ancho/alto en un elemento impar de trabajo."""
    alto, ancho = forma[:2]
    return tuple(2 * round((dimension * factor - 1) / 2) + 1
                 for dimension, factor in zip((ancho, alto), factores))


def detectar_patente(imagen: np.ndarray, metodo="D1", config=None, *, trazas=False) -> Deteccion:
    """Localiza una patente usando exclusivamente píxeles de la fotografía."""
    if metodo not in METODOS_DETECCION:
        raise ValueError(f"Método desconocido: {metodo}")
    config = config or ConfigDetector()
    gris = preparar_gris(imagen, config.max_lado)
    filtrada = (clahe(gris, config.clahe_limite, config.clahe_grilla) if metodo == "D3"
                else gaussiano(gris, config.gauss_kernel))
    base_densidad = gaussiano(gris, config.gauss_kernel) if metodo == "D3" else filtrada
    bordes = canny(base_densidad, config.canny_bajo, config.canny_alto, config.canny_kernel)
    if metodo in ("D0", "D1"):
        binaria, tamano = bordes, config.kernel_canny
    else:
        gx = sobel(filtrada, config.sobel_kernel)[0]
        binaria, tamano = otsu(intensidad_gradiente(gx)), config.kernel_sobel
        if config.kernel_relativo_sobel is not None:
            tamano = kernel_relativo(gris.shape, config.kernel_relativo_sobel)
    mascara = (binaria if metodo == "D0" else
               morfologia_rectangular(binaria, config.morfologia, tamano, config.iteraciones))
    if config.kernel_dilatacion is not None:
        mascara = morfologia_rectangular(
            mascara, "dilatacion", config.kernel_dilatacion, config.iteraciones_dilatacion
        )
    verticales = None
    if config.peso_vertical:
        gx, gy, _ = sobel(base_densidad, 3)
        verticales = (bordes > 0) & (np.abs(gx) > 1.25 * np.abs(gy))
    contornos, _ = cv2.findContours(mascara, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = extraer_candidatos(mascara, bordes, imagen.shape, config,
                                   contornos=contornos, bordes_verticales=verticales)
    mejor = seleccionar_candidato(candidatos, imagen, config)
    motivo = "" if mejor else ("sin_candidato_valido" if candidatos else "sin_contornos")
    pasos = {"gris": gris, "filtrada": filtrada, "binaria": binaria, "mascara": mascara} if trazas else {}
    return Deteccion(candidatos, mejor, motivo, pasos, len(contornos))


def recortar_rectangulo(imagen: np.ndarray, caja: Caja) -> np.ndarray:
    """Devuelve una copia de una caja válida, con extremos inferiores exclusivos."""
    alto, ancho = imagen.shape[:2]
    x0, y0, x1, y1 = caja
    if not (0 <= x0 < x1 <= ancho and 0 <= y0 < y1 <= alto):
        raise ValueError("La caja de recorte está fuera de la foto o es degenerada.")
    return imagen[y0:y1, x0:x1].copy()


def ordenar_vertices(vertices: np.ndarray) -> np.ndarray:
    """Ordena un cuadrilátero convexo como TL, TR, BR, BL de la placa horizontal."""
    puntos = np.asarray(vertices, dtype=np.float32)
    if puntos.shape != (4, 2) or not np.isfinite(puntos).all():
        raise ValueError("Se requieren cuatro vértices finitos.")
    casco = cv2.convexHull(puntos).reshape(-1, 2)
    if len(casco) != 4 or cv2.contourArea(casco) <= 1:
        raise ValueError("Cuadrilátero degenerado o no convexo.")
    delta = casco - casco.mean(axis=0)
    ciclo = casco[np.argsort(np.arctan2(delta[:, 1], delta[:, 0]))]
    lados = np.roll(ciclo, -1, axis=0) - ciclo
    largos = np.linalg.norm(lados, axis=1)
    par = max(range(2), key=lambda i: (largos[i] + largos[i + 2], abs(lados[i, 0])))
    borde = min((par, par + 2), key=lambda i: ciclo[i, 1] + ciclo[(i + 1) % 4, 1])
    return np.roll(ciclo, -borde, axis=0)


def rectificar_perspectiva(imagen: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Aplica una homografía a partir de cuatro esquinas válidas en la foto."""
    puntos = ordenar_vertices(vertices)
    alto, ancho = imagen.shape[:2]
    if not (np.all(puntos >= 0) and np.all(puntos < [ancho, alto])):
        raise ValueError("Las esquinas están fuera de la imagen.")
    lados = np.linalg.norm(np.roll(puntos, -1, axis=0) - puntos, axis=1)
    if min(lados) < 4 or max(lados[0], lados[2]) > 2 * min(lados[0], lados[2]) or (
        max(lados[1], lados[3]) > 2 * min(lados[1], lados[3])
    ):
        raise ValueError("La geometría del cuadrilátero es inestable.")
    w, h = round(max(lados[0], lados[2])) + 1, round(max(lados[1], lados[3])) + 1
    destino = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    matriz = cv2.getPerspectiveTransform(puntos, destino)
    return cv2.warpPerspective(imagen, matriz, (w, h))


def extraer_patente(imagen: np.ndarray, candidato: Candidato) -> Extraccion:
    """Conserva el recorte rectangular y rectifica cuando el contorno lo permite."""
    recorte = recortar_rectangulo(imagen, candidato.caja)
    vertices = candidato.poligono
    if vertices.shape != (4, 2) or not cv2.isContourConvex(vertices):
        return Extraccion(recorte, None, "sin_cuadrilatero_convexo")
    area = cv2.contourArea(candidato.rectangulo_rotado)
    if cv2.contourArea(vertices) < 0.72 * area:
        return Extraccion(recorte, None, "cuadrilatero_incompleto")
    if np.any(vertices <= 0) or np.any(vertices >= np.array(imagen.shape[1::-1]) - 1):
        return Extraccion(recorte, None, "cuadrilatero_toca_borde")
    try:
        rectificada = rectificar_perspectiva(imagen, vertices)
    except (ValueError, cv2.error):
        return Extraccion(recorte, None, "cuadrilatero_inestable")
    return Extraccion(recorte, rectificada, "rectificada")


def extraer_variante(imagen: np.ndarray, candidato: Candidato, modo: str) -> tuple[np.ndarray, str]:
    """Compara geometrías del mismo candidato y usa la caja si los lados son inestables."""
    rectangular = recortar_rectangulo(imagen, candidato.caja)
    if modo == "rectangular":
        return rectangular, modo
    if modo == "legado":
        extraccion = extraer_patente(imagen, candidato)
        if extraccion.rectificada is not None:
            return extraccion.rectificada, "perspectiva_legado"
        return rectangular, "rectangular"
    if modo not in ("rotado", "perspectiva"):
        raise ValueError(f"Geometría desconocida: {modo}.")
    puntos = candidato.rectangulo_rotado if modo == "rotado" else candidato.poligono
    try:
        ordenados = ordenar_vertices(puntos)
        lados = np.linalg.norm(np.roll(ordenados, -1, axis=0) - ordenados, axis=1)
        min_ancho, max_ancho = sorted(lados[::2])
        min_alto, max_alto = sorted(lados[1::2])
        if modo == "perspectiva" and (min_ancho / max_ancho < .78 or min_alto / max_alto < .72
                or cv2.contourArea(ordenados) < .84 * cv2.contourArea(candidato.rectangulo_rotado)):
            return rectangular, "rectangular"
        if not 1.45 <= max_ancho / max_alto <= 7:
            return rectangular, "rectangular"
        return rectificar_perspectiva(imagen, ordenados), modo
    except (ValueError, cv2.error):
        return rectangular, "rectangular"
