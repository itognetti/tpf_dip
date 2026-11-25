"""Preparación experimental de una región completa para Tesseract."""

from dataclasses import dataclass

import cv2
import numpy as np

from . import preprocessing as pre

FILTRADOS = ("M1", "M2", "M5", "otsu", "adaptativo", "high_boost", "morfologia")
INTERPOLACIONES = {"bicubica": cv2.INTER_CUBIC, "lineal": cv2.INTER_LINEAR}


@dataclass(frozen=True)
class PreparacionOCR:
    filtrado: str = "M2"
    factor: int = 1
    altura: int | None = None
    interpolacion: str = "bicubica"
    padding: int = 0
    polaridad: str = "original"
    recorte: str = "completo"

    def __post_init__(self):
        if self.factor not in (1, 2, 3, 4) or (
            self.altura is not None and (self.altura < 1 or self.factor != 1)
        ):
            raise ValueError("Indicá un factor 1–4 o una altura objetivo, sin combinarlos.")
        if self.padding < 0 or self.interpolacion not in INTERPOLACIONES or self.filtrado not in FILTRADOS:
            raise ValueError("Configuración espacial OCR inválida.")
        if self.polaridad not in ("original", "central", "borde") or self.recorte not in ("completo", "marco", "fila"):
            raise ValueError("Configuración de polaridad o recorte inválida.")


def reescalar(gris: np.ndarray, config: PreparacionOCR) -> np.ndarray:
    """Respeta la relación de aspecto y conserva el original cuando el factor es uno."""
    h, w = gris.shape
    alto = config.altura or h * config.factor
    if alto == h:
        return gris.copy()
    ancho = max(1, round(w * alto / h))
    return cv2.resize(gris, (ancho, alto), interpolation=INTERPOLACIONES[config.interpolacion])


def caja_texto(gris: np.ndarray, modo: str) -> tuple[int, int, int, int]:
    """Localiza una banda horizontal de energía, sin extraer caracteres individuales."""
    h, w = gris.shape
    if modo == "completo" or h < 20 or w < 30:
        return 0, 0, w, h
    gx = np.abs(cv2.Sobel(gris, cv2.CV_32F, 1, 0, ksize=3))
    centro = gx[:, w // 12:w - w // 12]
    perfil = centro.mean(axis=1)
    if modo == "marco":
        # Quita un margen solo si es uniforme en comparación con el área central.
        margen = max(1, round(h * 0.035))
        contraste = gris[h // 4:h - h // 4].std()
        arriba = margen if gris[:margen].std() < 0.35 * contraste else 0
        abajo = h - margen if gris[-margen:].std() < 0.35 * contraste else h
        return 0, arriba, w, abajo
    alto = max(1, round(h * 0.68))
    energia = np.convolve(perfil, np.ones(alto), mode="valid")
    inicio = int(energia.argmax())
    if energia[inicio] <= 0 or energia[inicio] < 0.78 * perfil.sum():
        return 0, 0, w, h
    seguridad = max(1, round(h * 0.035))
    return 0, max(0, inicio - seguridad), w, min(h, inicio + alto + seguridad)


def preparar_region(imagen: np.ndarray, config: PreparacionOCR) -> tuple[np.ndarray, dict]:
    """Produce el mismo uint8 gris que se guarda en ejemplos y se entrega al OCR."""
    gris = pre.escala_grises(imagen)
    x0, y0, x1, y1 = caja_texto(gris, config.recorte)
    gris = gris[y0:y1, x0:x1]
    invertida = False
    if config.polaridad == "borde":
        gris, invertida = pre.polaridad_ocr(gris)
    elif config.polaridad == "central":
        h, w = gris.shape
        mask = pre.otsu(gris)
        centro = mask[h//3:max(h//3+1, 4*h//5), w//8:max(w//8+1, 7*w//8)]
        invertida = bool(np.any(mask) and np.count_nonzero(centro) < centro.size / 2)
        if invertida:
            gris = pre.invertir(gris)

    if config.filtrado in ("M1", "M2", "M5"):
        if config.filtrado == "M2":
            gris = pre.gaussiano(gris)
        elif config.filtrado == "M5":
            gris = pre.clahe(gris)
        gris = reescalar(gris, config)
    else:
        gris = reescalar(gris, config)
        if config.filtrado == "adaptativo":
            gris = pre.umbral_adaptativo(pre.clahe(gris), bloque=31, constante=7)
        elif config.filtrado == "high_boost":
            gris = pre.otsu(pre.high_boost(gris))
        elif config.filtrado == "morfologia":
            gris = pre.limpiar_binaria_ocr(pre.otsu(gris))
        else:
            gris = pre.otsu(gris)
    p = config.padding
    if p:
        gris = cv2.copyMakeBorder(gris, p, p, p, p, cv2.BORDER_CONSTANT, value=255)
    return gris, {"caja_texto": (x0, y0, x1, y1), "polaridad_invertida": invertida}
