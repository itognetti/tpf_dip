"""Procesamiento clásico de una fotografía, independiente de las referencias."""

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pytesseract
from PIL import Image

from .detection import ConfigDetector, Deteccion, detectar_patente, extraer_variante
from .ocr import CARACTERES, ImagenInvalidaError, cargar_imagen, configurar_tesseract, reconocer_region
from .ocr_preparation import PreparacionOCR, preparar_region
from .segmentation import ConfigSegmentacion, Segmentacion, normalizar_caracter, segmentar

RAIZ = Path(__file__).resolve().parents[1]
CONFIGURACION = RAIZ / "config" / "pipeline_clasico.json"


@dataclass(frozen=True)
class ConfigPipeline:
    perfil: str
    detector: ConfigDetector
    geometria: str
    preparacion: PreparacionOCR
    psm: int
    segmentacion: ConfigSegmentacion

    def registro(self):
        return {
            "perfil_ocr": self.perfil,
            "detector": {"nombre": "B3_FILTRO", "metodo": "D3", "parametros": asdict(self.detector)},
            "geometria": self.geometria,
            "ocr": {
                "psm": self.psm, "idioma": "eng", "alfabeto": CARACTERES,
                "parametros": asdict(self.preparacion),
                "timeout_segundos": 20, "omp_thread_limit": os.environ.get("OMP_THREAD_LIMIT"),
            },
            "segmentacion": {"metodo": "S0", "banda": "fila", "parametros": asdict(self.segmentacion)},
        }


def cargar_configuracion(perfil="exactitud"):
    """Lee los parámetros congelados de 4B y 5, disponibles sin los experimentos."""
    datos = json.loads(CONFIGURACION.read_text(encoding="utf-8"))
    ocr = datos["perfiles"][perfil]
    return ConfigPipeline(perfil, ConfigDetector(**datos["detector"]["config"]),
                          datos["geometria"], PreparacionOCR(**ocr["preparacion"]),
                          ocr["psm"], ConfigSegmentacion(**datos["segmentacion"]))


def preparar_motor():
    """Comprueba ejecutable e idioma una vez, indicando por qué el OCR no está disponible."""
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    local = Path(sys.prefix) / "Tesseract-OCR" / "tesseract.exe"
    if not os.environ.get("TESSERACT_CMD") and local.is_file():
        os.environ["TESSERACT_CMD"] = str(local)
    configurar_tesseract()
    try:
        version = str(pytesseract.get_tesseract_version())
        if "eng" not in pytesseract.get_languages():
            return version, "Tesseract no encuentra los datos del idioma eng."
    except pytesseract.TesseractNotFoundError:
        return None, "Tesseract no disponible. Revisá TESSERACT_CMD o PATH."
    except (OSError, RuntimeError) as error:
        return None, f"No se pudo iniciar Tesseract: {error}"
    return version, ""


@dataclass
class ResultadoPipeline:
    fila: dict
    foto: np.ndarray | None = None
    deteccion: Deteccion | None = None
    recorte: np.ndarray | None = None
    entrada_ocr: np.ndarray | None = None
    segmentacion: Segmentacion | None = None
    caracteres: list = field(default_factory=list)


def procesar_foto(ruta: Path, config: ConfigPipeline, problema_motor="") -> ResultadoPipeline:
    """Procesa una foto y conserva el estado de cada etapa, aun si falla otra rama."""
    inicio = perf_counter()
    fila = {
        "archivo": ruta.name, "estado_deteccion": "no_ejecutada", "caja_detectada": None,
        "detector": "B3_FILTRO", "geometria_efectiva": None, "preprocesamiento": config.preparacion.filtrado,
        "perfil_ocr": config.perfil, "psm": config.psm, "estado_ocr": "no_ejecutado",
        "caja_texto": None, "polaridad_invertida": None, "texto_literal": "", "texto_normalizado": "",
        "formato_valido": None, "confianza": None, "estado_segmentacion": "no_ejecutada",
        "regiones_segmentadas": None, "estado_normalizacion": "no_ejecutada",
        "caracteres_normalizados": None, "ms_procesamiento": None, "motivo_fallo": "",
    }
    salida = ResultadoPipeline(fila)
    fallos = []

    def terminar():
        fila["ms_procesamiento"] = round((perf_counter() - inicio) * 1000, 3)
        fila["motivo_fallo"] = " | ".join(fallos)
        return salida

    try:
        salida.foto = np.asarray(cargar_imagen(ruta))
    except (OSError, ImagenInvalidaError) as error:
        fila["estado_deteccion"] = "imagen_invalida"
        fallos.append(str(error))
        return terminar()

    try:
        salida.deteccion = detectar_patente(salida.foto, "D3", config.detector, trazas=True)
    except (ValueError, cv2.error) as error:
        fila["estado_deteccion"] = "error_deteccion"
        fallos.append(f"Detección: {error}")
        return terminar()

    candidato = salida.deteccion.seleccionado
    if candidato is None:
        fila["estado_deteccion"] = "patente_no_detectada"
        fallos.append(salida.deteccion.motivo)
        return terminar()

    fila.update(estado_deteccion="patente_detectada", caja_detectada=candidato.caja)
    try:
        salida.recorte, fila["geometria_efectiva"] = extraer_variante(salida.foto, candidato, config.geometria)
    except (ValueError, cv2.error) as error:
        fallos.append(f"Extracción: {error}")
        return terminar()

    try:
        salida.entrada_ocr, opciones = preparar_region(salida.recorte, config.preparacion)
        fila.update(opciones)
    except (ValueError, cv2.error) as error:
        fila["estado_ocr"] = "error_preparacion"
        fallos.append(f"Preparación OCR: {error}")
    else:
        if problema_motor:
            fila["estado_ocr"] = "motor_no_disponible"
            fallos.append(problema_motor)
        else:
            try:
                lectura = reconocer_region(Image.fromarray(salida.entrada_ocr), config.psm)
            except pytesseract.TesseractNotFoundError as error:
                fila["estado_ocr"] = "motor_no_disponible"
                fallos.append(str(error))
            except (OSError, RuntimeError, ValueError) as error:
                fila["estado_ocr"] = "error_ocr"
                fallos.append(f"OCR: {error}")
            else:
                if not lectura.texto_normalizado:
                    estado = "sin_texto"
                elif lectura.formato_valido:
                    estado = "formato_valido"
                else:
                    estado = "formato_invalido"
                fila.update(estado_ocr=estado, texto_literal=lectura.texto_original,
                            texto_normalizado=lectura.texto_normalizado,
                            formato_valido=lectura.formato_valido, confianza=lectura.confianza)

    try:
        salida.segmentacion = segmentar(salida.recorte, "S0", "fila", config.segmentacion)
    except (ValueError, cv2.error) as error:
        fila["estado_segmentacion"] = "error_segmentacion"
        fallos.append(f"Segmentación: {error}")
        return terminar()

    regiones = salida.segmentacion
    fila["regiones_segmentadas"] = len(regiones.cajas_trabajo)
    if not regiones.cajas_trabajo:
        fila["estado_segmentacion"] = "sin_regiones"
        fallos.append("La binarización y el filtrado no entregaron regiones candidatas.")
        return terminar()

    fila.update(estado_segmentacion="regiones_candidatas", caracteres_normalizados=0)
    for indice, (x0, y0, x1, y1) in enumerate(regiones.cajas_trabajo, 1):
        mascara = regiones.binaria[y0:y1, x0:x1]
        try:
            normalizado = normalizar_caracter(mascara, config.segmentacion.lado_interno, polaridad="claro")
        except (ValueError, cv2.error) as error:
            fallos.append(f"Normalización de región {indice}: {error}")
            normalizado = None
        salida.caracteres.append((indice, mascara, normalizado))
    cantidad = sum(normalizado is not None for _, _, normalizado in salida.caracteres)
    estado = "completada" if cantidad == len(salida.caracteres) else ("parcial" if cantidad else "error_normalizacion")
    fila.update(estado_normalizacion=estado, caracteres_normalizados=cantidad)
    return terminar()
