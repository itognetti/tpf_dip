"""Segmentación espacial de una patente y preparación de entradas para la BPN."""

from dataclasses import dataclass
from math import ceil, floor

import cv2
import numpy as np

from . import preprocessing as pre
from .detection import Caja
from .ocr_preparation import PreparacionOCR, caja_texto, reescalar

METODOS = ("S0", "S1", "S2")
BANDAS = ("completa", "marco", "fila")
CLASES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class ConfigSegmentacion:
    altura: int = 96
    clahe: float = 1.5
    bloque_adaptativo: int = 25
    constante_adaptativa: float = 7
    quitar_marcos: bool = True
    altura_min: float = .32
    altura_max: float = .98
    ancho_rel_max: float = .35
    area_rel_min: float = .001
    aspecto_min: float = .035
    aspecto_max: float = 1.08
    densidad_min: float = .11
    desalineacion_max: float = .23
    proyeccion_min: float = .25
    unir_fragmentos: bool = True
    separar_unidos: bool = True
    lado_interno: int = 20

    def __post_init__(self):
        if self.altura < 16 or self.bloque_adaptativo < 3 or self.bloque_adaptativo % 2 != 1:
            raise ValueError("Revisá la altura de trabajo y el bloque adaptativo impar.")
        if not 0 < self.altura_min < self.altura_max <= 1 or not 0 < self.aspecto_min < self.aspecto_max:
            raise ValueError("Los intervalos de altura o aspecto son inválidos.")
        if not all(0 < x <= 1 for x in (self.ancho_rel_max, self.area_rel_min, self.densidad_min,
                                         self.desalineacion_max, self.proyeccion_min)):
            raise ValueError("Las proporciones deben estar entre cero y uno.")
        if self.clahe < 0 or not 1 <= self.lado_interno <= 28:
            raise ValueError("Revisá CLAHE y el lado interno de la normalización.")


@dataclass
class Segmentacion:
    banda: Caja
    binaria: np.ndarray
    perfil: np.ndarray
    cajas: list[Caja]
    cajas_trabajo: list[Caja]
    clara: bool
    uniones: int
    separaciones: int


@dataclass
class CaracterNormalizado:
    ajuste: np.ndarray
    escalada: np.ndarray
    lienzo: np.ndarray
    imagen: np.ndarray
    vector: np.ndarray
    centro_masa: tuple[float, float]


def binarizar_caracteres(gris: np.ndarray, metodo: str, config: ConfigSegmentacion):
    """Devuelve trazos blancos. Estima la polaridad en el interior por Otsu."""
    if not gris.size or int(gris.min()) == int(gris.max()):
        return np.zeros_like(gris), False
    otsu = pre.otsu(gris)
    h, w = gris.shape
    centro = otsu[h // 4:h - h // 6, w // 10:w - w // 10]
    clara = np.count_nonzero(centro) < centro.size / 2
    if metodo == "S1":
        caracteres_oscuros = pre.invertir(gris) if clara else gris
        return pre.invertir(pre.umbral_adaptativo(
            caracteres_oscuros, config.bloque_adaptativo, config.constante_adaptativa
        )), clara
    return (otsu if clara else pre.invertir(otsu)), clara


def quitar_lineas_perifericas(binaria: np.ndarray) -> np.ndarray:
    """Borra segmentos horizontales muy largos solo cerca de los bordes."""
    h, w = binaria.shape
    longitud = max(3, round(.65 * w))
    if longitud % 2 == 0:
        longitud += 1
    elemento = cv2.getStructuringElement(cv2.MORPH_RECT, (longitud, 1))
    lineas = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, elemento)
    borde = max(1, round(.14 * h))
    lineas[borde:h - borde] = 0
    return cv2.subtract(binaria, lineas)


def _caja_rect(rectangulo) -> Caja:
    x, y, w, h = map(int, rectangulo)
    return x, y, x + w, y + h


def cajas_contornos(binaria):
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [_caja_rect(cv2.boundingRect(c)) for c in contornos]


def cajas_componentes(binaria):
    _, _, datos, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    return [_caja_rect(fila[:4]) for fila in datos[1:]]


def _ajustar(binaria, caja):
    x0, y0, x1, y1 = caja
    region = binaria[y0:y1, x0:x1]
    if not region.size or cv2.countNonZero(region) == 0:
        return None
    x, y, w, h = cv2.boundingRect(region)
    return x0 + x, y0 + y, x0 + x + w, y0 + y + h


def cajas_proyeccion(binaria, minimo=.25):
    perfil = np.count_nonzero(binaria, axis=0)
    activas = perfil >= max(1, ceil(minimo * binaria.shape[0]))
    cambios = np.flatnonzero(np.diff(np.r_[False, activas, False]))
    cajas = [_ajustar(binaria, (int(i), 0, int(j), binaria.shape[0]))
             for i, j in zip(cambios[::2], cambios[1::2])]
    return [caja for caja in cajas if caja is not None]


def _dim(caja):
    return caja[2] - caja[0], caja[3] - caja[1]


def unir_fragmentos(cajas, alto_banda):
    """Une piezas verticales próximas y con solapamiento horizontal fuerte."""
    salida = []
    cantidad = 0
    for caja in sorted(cajas, key=lambda c: (c[1], c[0])):
        ancho, alto = _dim(caja)
        if alto < .08 * alto_banda:
            salida.append(caja)
            continue
        for i, otra in enumerate(salida):
            ancho_otro, alto_otro = _dim(otra)
            solape = min(caja[2], otra[2]) - max(caja[0], otra[0])
            hueco = caja[1] - otra[3]
            total = (min(caja[0], otra[0]), otra[1], max(caja[2], otra[2]), caja[3])
            w, h = _dim(total)
            if (alto_otro >= .08 * alto_banda and solape >= .72 * min(ancho, ancho_otro)
                    and 0 <= hueco <= .10 * alto_banda and h <= .96 * alto_banda and w <= h):
                salida[i] = total
                cantidad += 1
                break
        else:
            salida.append(caja)
    return salida, cantidad


def separar_unidos(binaria, caja):
    """Divide una caja ancha solo si hay un valle interior con dos trazos altos."""
    ancho, alto = _dim(caja)
    if ancho < 8 or ancho < .96 * alto:
        return [caja]
    x0, y0, x1, y1 = caja
    perfil = np.count_nonzero(binaria[y0:y1, x0:x1], axis=0)
    a, b = floor(.30 * ancho), ceil(.70 * ancho)
    corte = int(perfil[a:b].argmin()) + a
    if perfil[corte] > .22 * perfil.max():
        return [caja]
    partes = [_ajustar(binaria, c) for c in ((x0, y0, x0 + corte, y1), (x0 + corte, y0, x1, y1))]
    if all(p is not None and _dim(p)[1] >= .68 * alto for p in partes):
        return partes
    return [caja]


def filtrar_candidatos(binaria, cajas, config):
    h, w = binaria.shape
    salida = []
    for caja in cajas:
        ancho, alto = _dim(caja)
        x0, y0, x1, y1 = caja
        if not (config.altura_min * h <= alto <= config.altura_max * h
                and 2 <= ancho <= config.ancho_rel_max * w
                and config.aspecto_min <= ancho / alto <= config.aspecto_max):
            continue
        pixeles = cv2.countNonZero(binaria[y0:y1, x0:x1])
        if pixeles >= config.area_rel_min * h * w and pixeles / (ancho * alto) >= config.densidad_min:
            salida.append(caja)
    if len(salida) >= 3:
        centro = np.median([(c[1] + c[3]) / 2 for c in salida])
        salida = [c for c in salida if abs((c[1] + c[3]) / 2 - centro) <= config.desalineacion_max * h]
    unicas = []
    for c in sorted(salida, key=lambda a: (-a[3] + a[1], a[0])):
        w_c, h_c = _dim(c)
        superpuesta = any(
            max(0, min(c[2], a[2]) - max(c[0], a[0])) * max(0, min(c[3], a[3]) - max(c[1], a[1]))
            >= .65 * min(w_c * h_c, _dim(a)[0] * _dim(a)[1]) for a in unicas
        )
        if not superpuesta:
            unicas.append(c)
    return sorted(unicas, key=lambda c: (c[0], c[1]))


def segmentar(imagen: np.ndarray, metodo="S0", banda="completa", config=None) -> Segmentacion:
    """Segmenta una región de patente únicamente con sus píxeles."""
    if metodo not in METODOS or banda not in BANDAS:
        raise ValueError("Método o banda de segmentación desconocidos.")
    config = config or ConfigSegmentacion()
    gris = pre.escala_grises(imagen)
    region = caja_texto(gris, "completo" if banda == "completa" else banda)
    x0, y0, x1, y1 = region
    gris = reescalar(gris[y0:y1, x0:x1], PreparacionOCR(altura=config.altura))
    if config.clahe:
        gris = pre.clahe(gris, config.clahe, (4, 4))
    binaria, clara = binarizar_caracteres(gris, metodo, config)
    if config.quitar_marcos:
        binaria = quitar_lineas_perifericas(binaria)
    if metodo == "S0":
        cajas = cajas_contornos(binaria)
    elif metodo == "S1":
        cajas = cajas_componentes(binaria)
    else:
        cajas = cajas_proyeccion(binaria, config.proyeccion_min)
    uniones = separaciones = 0
    if config.unir_fragmentos:
        cajas, uniones = unir_fragmentos(cajas, binaria.shape[0])
    if config.separar_unidos:
        partes = [p for caja in cajas for p in separar_unidos(binaria, caja)]
        separaciones = len(partes) - len(cajas)
        cajas = partes
    cajas = filtrar_candidatos(binaria, cajas, config)
    sx, sy = (x1 - x0) / binaria.shape[1], (y1 - y0) / binaria.shape[0]
    cajas_original = [(x0 + floor(a * sx), y0 + floor(b * sy),
                       min(x1, x0 + ceil(c * sx)), min(y1, y0 + ceil(d * sy))) for a, b, c, d in cajas]
    return Segmentacion(region, binaria, np.count_nonzero(binaria, axis=0), cajas_original,
                        cajas, clara, uniones, separaciones)


def normalizar_caracter(imagen: np.ndarray, lado_interno=20, polaridad="auto") -> CaracterNormalizado:
    """Ajusta y centra un carácter impreso siguiendo el procedimiento 20→28 del profesor."""
    if not isinstance(lado_interno, int) or not 1 <= lado_interno <= 28:
        raise ValueError("El lado interno debe ser un entero entre 1 y 28.")
    gris = pre.escala_grises(imagen)
    if polaridad not in ("auto", "claro", "oscuro"):
        raise ValueError("Polaridad de carácter desconocida.")
    if polaridad == "auto":
        if gris.max() == gris.min():
            raise ValueError("El recorte no contiene contraste de carácter.")
        gris, _ = pre.polaridad_ocr(gris)
        polaridad = "oscuro"
    trazo = pre.otsu(gris)
    if polaridad == "oscuro":
        trazo = pre.invertir(trazo)
    caja = _ajustar(trazo, (0, 0, trazo.shape[1], trazo.shape[0]))
    if caja is None:
        raise ValueError("El recorte no contiene trazos.")
    x0, y0, x1, y1 = caja
    ajuste = trazo[y0:y1, x0:x1]
    ancho, alto = _dim(caja)
    escala = lado_interno / max(ancho, alto)
    w, h = max(1, round(ancho * escala)), max(1, round(alto * escala))
    # Ajustar a 20 preserva el área y la interpolación conserva niveles útiles.
    escalada = cv2.resize(ajuste.astype(np.float32), (w, h),
                          interpolation=cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR)
    # Si hay una contribución positiva, conserva al menos un nivel de intensidad.
    escalada = np.maximum(np.rint(escalada), escalada > 0).clip(0, 255).astype(np.uint8)
    momentos = cv2.moments(escalada)
    cx, cy = momentos["m10"] / momentos["m00"], momentos["m01"] / momentos["m00"]
    x, y = min(28 - w, max(0, round(14 - cx))), min(28 - h, max(0, round(14 - cy)))
    lienzo = np.zeros((28, 28), dtype=np.uint8)
    lienzo[y:y + h, x:x + w] = escalada
    valores = lienzo.astype(np.float32) / np.float32(255)
    return CaracterNormalizado(ajuste, escalada, lienzo, valores, valores.reshape(784), (x + cx, y + cy))
