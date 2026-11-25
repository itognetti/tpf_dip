"""OCR básico sobre una región de patente recortada manualmente."""

import csv
import os
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from statistics import mean

from PIL import Image

CARACTERES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MODOS_PSM = (7, 8)
MODOS_PSM_REGION = (6, 7, 8, 13)
PATRON_PATENTE = re.compile(r"(?:[A-Z]{3}[0-9]{3}|[A-Z]{2}[0-9]{3}[A-Z]{2})")


class ImagenInvalidaError(ValueError):
    """El archivo no contiene una imagen que Pillow pueda decodificar."""


@dataclass(frozen=True)
class ResultadoOCR:
    psm: int
    texto_original: str
    texto_normalizado: str
    confianza: float | None
    formato_valido: bool


def cargar_imagen(ruta: str | Path) -> Image.Image:
    """Carga una imagen RGB, desconectada del archivo y lista para OCR."""
    ruta = Path(ruta)
    if ruta.is_dir():
        raise IsADirectoryError(f"Se esperaba un archivo de imagen: {ruta}")

    # Abrir primero el archivo permite conservar los errores del sistema.
    with ruta.open("rb") as archivo:
        try:
            with Image.open(archivo) as imagen:
                if "A" in imagen.getbands() or "transparency" in imagen.info:
                    capa = imagen.convert("RGBA")
                    fondo = Image.new("RGBA", capa.size, "white")
                    return Image.alpha_composite(fondo, capa).convert("RGB")
                return imagen.convert("RGB")
        except (OSError, Image.DecompressionBombError) as error:
            raise ImagenInvalidaError(f"No se pudo decodificar la imagen: {ruta}") from error


def normalizar_texto(texto: str) -> str:
    """Conserva letras ASCII y dígitos, y pasa las letras a mayúsculas."""
    return re.sub(r"[^a-zA-Z0-9]", "", texto).upper()


def es_formato_valido(texto_normalizado: str) -> bool:
    """Valida el texto completo ya normalizado contra ambos formatos."""
    return PATRON_PATENTE.fullmatch(texto_normalizado) is not None


def configurar_tesseract() -> None:
    """Usa TESSERACT_CMD si está definida, o busca tesseract en PATH."""
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = os.environ.get("TESSERACT_CMD") or "tesseract"


def _confianza_media(tsv: str) -> float | None:
    """Promedio de las confianzas disponibles en las filas de palabra."""
    confianzas = []
    for fila in csv.DictReader(StringIO(tsv), delimiter="\t"):
        if fila["level"] == "5" and fila["text"].strip():
            confianza = float(fila["conf"])
            if 0 <= confianza <= 100:
                confianzas.append(confianza)
    return mean(confianzas) if confianzas else None


def reconocer_patente(imagen: Image.Image, psm: int = 7) -> ResultadoOCR:
    """Lee texto y confianza de un recorte sin corregir la predicción."""
    if not isinstance(psm, int) or psm not in MODOS_PSM:
        raise ValueError(f"PSM {psm} no admitido. Modos disponibles: {MODOS_PSM}.")

    import pytesseract

    configurar_tesseract()
    configuracion = f"--psm {psm} -c tessedit_char_whitelist={CARACTERES}"
    original = pytesseract.image_to_string(imagen, lang="eng", config=configuracion)
    tsv = pytesseract.image_to_data(imagen, lang="eng", config=configuracion)
    return _resultado(psm, original, tsv)


def reconocer_region(imagen: Image.Image, psm: int = 7) -> ResultadoOCR:
    """Obtiene TXT y TSV en una sola llamada a Tesseract para los ensayos 4B."""
    if not isinstance(psm, int) or psm not in MODOS_PSM_REGION:
        raise ValueError(f"PSM de región inválido: {psm}.")

    from pytesseract import pytesseract as motor

    configurar_tesseract()
    config = (f"--psm {psm} -c tessedit_char_whitelist={CARACTERES} "
              "-c tessedit_create_txt=1 -c tessedit_create_tsv=1")
    with motor.save(imagen) as (base, entrada):
        motor.run_tesseract(
            input_filename=entrada, output_filename_base=base, extension="txt",
            lang="eng", config=config, timeout=20,
        )
        original = Path(f"{base}.txt").read_bytes().decode("utf-8")
        tsv = Path(f"{base}.tsv").read_bytes().decode("utf-8")
    return _resultado(psm, original, tsv)


def _resultado(psm, original, tsv):
    normalizado = normalizar_texto(original)
    return ResultadoOCR(
        psm=psm,
        texto_original=original,
        texto_normalizado=normalizado,
        confianza=_confianza_media(tsv),
        formato_valido=es_formato_valido(normalizado),
    )
