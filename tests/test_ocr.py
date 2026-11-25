import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from src.ocr import (
    CARACTERES,
    MODOS_PSM,
    ImagenInvalidaError,
    cargar_imagen,
    configurar_tesseract,
    es_formato_valido,
    normalizar_texto,
    reconocer_patente,
)


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [(" ab-123 cd\n", "AB123CD"), (" ao c12s\n", "AOC12S"), ("Ñéß１２!", "")],
)
def test_normalizar_texto(texto, esperado):
    assert normalizar_texto(texto) == esperado


@pytest.mark.parametrize("texto", ["ABC123", "AB123CD", "ZZZ000", "ZZ000ZZ"])
def test_formatos_validos(texto):
    assert es_formato_valido(texto)


@pytest.mark.parametrize(
    "texto", ["", "abc123", " ABC123", "ABC123\n", "A1C123", "AB12SCD", "AB１２３CD"]
)
def test_formatos_invalidos(texto):
    assert not es_formato_valido(texto)


def test_cargar_imagen_rgb_con_ruta_unicode(tmp_path):
    ruta = tmp_path / "paténte.png"
    Image.new("L", (24, 8), 128).save(ruta)
    imagen = cargar_imagen(ruta)
    ruta.unlink()
    assert imagen.mode == "RGB"
    assert imagen.size == (24, 8)
    assert imagen.getpixel((0, 0)) == (128, 128, 128)


def test_cargar_imagen_transparente_sobre_blanco(tmp_path):
    ruta = tmp_path / "transparente.png"
    imagen = Image.new("RGBA", (2, 1), (0, 0, 0, 0))
    imagen.putpixel((1, 0), (0, 0, 0, 255))
    imagen.save(ruta)
    cargada = cargar_imagen(ruta)
    assert cargada.getpixel((0, 0)) == (255, 255, 255)
    assert cargada.getpixel((1, 0)) == (0, 0, 0)


def test_cargar_archivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        cargar_imagen(tmp_path / "ausente.png")


def test_cargar_directorio(tmp_path):
    with pytest.raises(IsADirectoryError):
        cargar_imagen(tmp_path)


def test_cargar_archivo_invalido(tmp_path):
    ruta = tmp_path / "texto.png"
    ruta.write_text("archivo de texto", encoding="utf-8")
    with pytest.raises(ImagenInvalidaError, match="No se pudo decodificar"):
        cargar_imagen(ruta)


def test_cargar_imagen_truncada(tmp_path):
    ruta = tmp_path / "truncada.png"
    Image.new("RGB", (24, 8), "white").save(ruta)
    ruta.write_bytes(ruta.read_bytes()[:50])
    with pytest.raises(ImagenInvalidaError):
        cargar_imagen(ruta)


def test_cargar_archivo_sin_permiso(tmp_path, monkeypatch):
    ruta = tmp_path / "restringida.png"
    ruta.touch()

    def sin_permiso(*args, **kwargs):
        raise PermissionError("Acceso denegado")

    monkeypatch.setattr(Path, "open", sin_permiso)
    with pytest.raises(PermissionError, match="Acceso denegado"):
        cargar_imagen(ruta)


@pytest.mark.parametrize("psm", [6, 7.0, "7"])
def test_psm_invalido(psm):
    with pytest.raises(ValueError, match="PSM"):
        reconocer_patente(Image.new("RGB", (24, 8)), psm=psm)


@pytest.mark.parametrize(
    ("texto", "confianzas", "esperado"),
    [(" ABC-12S\n", ["-1", "80.5", "100", "0", "-1"], 40.25), ("", ["-1"] * 5, None)],
)
def test_reconocer_con_resultados_simulados(monkeypatch, texto, confianzas, esperado):
    llamadas = []

    def devolver_texto(imagen, **kwargs):
        llamadas.append(kwargs)
        return texto

    def devolver_datos(imagen, **kwargs):
        llamadas.append(kwargs)
        renglones = zip([1, 5, 5, 5, 5], ["", "ABC-12S", " ", "AB", "CD"], confianzas)
        return "level\tconf\ttext\n" + "".join(
            f"{nivel}\t{valor}\t{palabra}\n" for nivel, palabra, valor in renglones
        )

    motor = SimpleNamespace(tesseract_cmd="")
    biblioteca = SimpleNamespace(
        pytesseract=motor,
        image_to_string=devolver_texto,
        image_to_data=devolver_datos,
    )
    monkeypatch.setitem(sys.modules, "pytesseract", biblioteca)
    monkeypatch.setenv("TESSERACT_CMD", "ruta con espacios/tesseract.exe")
    resultado = reconocer_patente(Image.new("RGB", (24, 8)), psm=8)

    configuracion = f"--psm 8 -c tessedit_char_whitelist={CARACTERES}"
    assert llamadas == [
        {"lang": "eng", "config": configuracion},
        {"lang": "eng", "config": configuracion},
    ]
    assert motor.tesseract_cmd == "ruta con espacios/tesseract.exe"
    assert resultado.psm == 8
    assert resultado.texto_original == texto
    assert resultado.texto_normalizado == ("ABC12S" if texto else "")
    assert resultado.confianza == esperado
    assert not resultado.formato_valido


@pytest.fixture(scope="module")
def tesseract_disponible():
    pytesseract = pytest.importorskip("pytesseract")
    configurar_tesseract()
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        pytest.skip("Motor Tesseract no disponible en PATH ni en TESSERACT_CMD")
    if "eng" not in pytesseract.get_languages():
        pytest.skip("Tesseract no tiene datos para el idioma eng")


@pytest.mark.parametrize("psm", MODOS_PSM)
@pytest.mark.parametrize(
    ("archivo", "etiqueta"),
    [("demo_antigua_ABC123.png", "ABC123"), ("demo_mercosur_AB123CD.png", "AB123CD")],
)
def test_ocr_con_tesseract(tesseract_disponible, psm, archivo, etiqueta):
    ruta = Path(__file__).resolve().parents[1] / "data" / "examples" / archivo
    resultado = reconocer_patente(cargar_imagen(ruta), psm=psm)

    assert resultado.texto_normalizado == etiqueta
    assert resultado.texto_original.strip()
    assert resultado.formato_valido
    assert resultado.confianza is not None
    assert 0 <= resultado.confianza <= 100
