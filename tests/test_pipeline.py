import csv
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
import pytesseract
from PIL import Image

import main
from scripts.run_detection import casos_localizacion
from src import pipeline
from src.ocr import ResultadoOCR
from src.segmentation import Segmentacion


def _foto_sintetica():
    casos, _ = casos_localizacion(pipeline.RAIZ / "data", sintetico=True)
    return casos[0]["sintetica"]


@pytest.fixture
def foto(tmp_path):
    ruta = tmp_path / "sintética.png"
    Image.fromarray(_foto_sintetica()).save(ruta)
    return ruta


def test_configuraciones_congeladas():
    carpeta = pipeline.RAIZ / "results"
    archivo = carpeta / "stage4b/config.json"
    if not archivo.is_file():
        pytest.skip("Configuración histórica local no disponible")
    etapa4b = json.loads(archivo.read_text(encoding="utf-8"))
    for perfil, codigo in [("exactitud", "B0"), ("cer", "PSM6")]:
        cfg = pipeline.cargar_configuracion(perfil)
        registro = json.loads(json.dumps(cfg.registro()))
        assert registro["detector"]["parametros"] == etapa4b["detector"]["variantes"]["B3_FILTRO"]["config"]
        assert registro["geometria"] == etapa4b["geometria_seleccionada"] == "rotado"
        assert registro["ocr"]["parametros"] == etapa4b["ocr"]["configuraciones"][codigo]["preparacion"]
        assert registro["ocr"]["psm"] == etapa4b["ocr"]["configuraciones"][codigo]["psm"]


@pytest.mark.parametrize("texto, estado, valido", [
    ("ABC123", "formato_valido", True), ("A2", "formato_invalido", False), ("", "sin_texto", False),
])
def test_pipeline_independiente_de_xml_y_etiquetas(foto, monkeypatch, texto, estado, valido):
    original = foto.read_bytes()
    leer = Path.open

    def sin_referencias(ruta, *args, **kwargs):
        assert ruta.suffix.lower() != ".xml" and ruta.name not in {"ground_truth.csv", "character_annotations.csv"}
        return leer(ruta, *args, **kwargs)

    def sin_xml(*args, **kwargs):
        raise AssertionError("Intento de consultar XML")

    monkeypatch.setattr(Path, "open", sin_referencias)
    monkeypatch.setattr(ElementTree, "parse", sin_xml)
    monkeypatch.setattr(pipeline, "reconocer_region", lambda im, psm: ResultadoOCR(psm, texto+"\n", texto, 12.5, valido))
    resultado = pipeline.procesar_foto(foto, pipeline.cargar_configuracion())
    fila = resultado.fila
    assert fila["estado_deteccion"] == "patente_detectada"
    assert fila["estado_ocr"] == estado and fila["texto_literal"] == texto + "\n" and fila["confianza"] == 12.5
    assert fila["estado_segmentacion"] == "regiones_candidatas"
    assert fila["estado_normalizacion"] == "completada"
    assert fila["caracteres_normalizados"] == len(resultado.caracteres) > 0
    assert all(c.imagen.shape == (28, 28) and c.vector.shape == (784,) for _, _, c in resultado.caracteres)
    assert np.array_equal(resultado.foto, _foto_sintetica())
    assert foto.read_bytes() == original


def test_carpeta_exporta_y_continua_con_motor_ausente(foto, monkeypatch):
    entrada = foto.parent
    (entrada / "a_rota.jpg").write_bytes(b"archivo roto")
    Image.new("RGB", (120, 80)).save(entrada / "b_vacia.png")
    (entrada / "ignorar.txt").write_text("no es una foto")
    salida = entrada / "salida"
    monkeypatch.setenv("TESSERACT_CMD", str(entrada / "inexistente.exe"))
    assert main.main(["--entrada", str(entrada), "--salida", str(salida), "--perfil", "cer"]) == 0
    with (salida / "resultados.csv").open(encoding="utf-8-sig", newline="") as archivo:
        rota, vacia, valida = list(csv.DictReader(archivo))
    assert rota["estado_deteccion"] == "imagen_invalida" and rota["motivo_fallo"]
    assert vacia["estado_deteccion"] == "patente_no_detectada" and not vacia["caja_detectada"]
    assert valida["estado_ocr"] == "motor_no_disponible" and valida["caracteres_normalizados"]
    assert all((salida / f["ruta_visualizations"]).is_file() for f in (rota, vacia, valida))
    assert all((salida / valida[f"ruta_{d}"]).is_file() for d in ("detected", "cropped", "processed"))
    normalizados = list((salida / "normalized").glob("*.png"))
    assert normalizados
    with Image.open(normalizados[0]) as imagen:
        assert imagen.mode == "L" and imagen.size == (28, 28) and imagen.getbbox()
    assert "motor_no_disponible" in (salida / "resultados.txt").read_text(encoding="utf-8")
    config = json.loads((salida / "config.json").read_text(encoding="utf-8"))
    assert config["ocr"]["psm"] == 6 and config["problema_motor"]
    assert {"accuracy", "cer", "iou"}.isdisjoint(valida)


def test_error_ocr_y_segmentacion_con_estados_independientes(foto, monkeypatch):
    def romper(*args, **kwargs):
        raise pytesseract.TesseractError(2, "fallo simulado")

    corte = np.zeros((32, 120), np.uint8)
    monkeypatch.setattr(pipeline, "reconocer_region", romper)
    monkeypatch.setattr(pipeline, "segmentar", lambda *a: Segmentacion((0, 0, 120, 32), corte, corte.sum(0),
                                                                        [], [], False, 0, 0))
    resultado = pipeline.procesar_foto(foto, pipeline.cargar_configuracion())
    assert resultado.fila["estado_ocr"] == "error_ocr"
    assert resultado.fila["estado_segmentacion"] == "sin_regiones"
    assert resultado.fila["regiones_segmentadas"] == 0 and resultado.fila["caracteres_normalizados"] is None
    assert resultado.fila["motivo_fallo"] and resultado.entrada_ocr is not None


def test_pipeline_con_tesseract_real(foto, monkeypatch):
    with monkeypatch.context() as entorno:
        entorno.setenv("TESSERACT_CMD", pipeline.os.environ.get("TESSERACT_CMD", "tesseract"))
        _, problema = pipeline.preparar_motor()
        if problema:
            pytest.skip(problema)
        resultado = pipeline.procesar_foto(foto, pipeline.cargar_configuracion())
    assert resultado.fila["estado_ocr"] in {"formato_valido", "formato_invalido", "sin_texto"}
    assert resultado.fila["estado_normalizacion"] == "completada"


def test_cli_protege_una_foto_en_el_directorio_de_exportacion(tmp_path):
    carpeta = tmp_path / "cropped"
    carpeta.mkdir()
    foto = carpeta / "conservar.png"
    Image.new("RGB", (20, 20)).save(foto)
    original = foto.read_bytes()
    with pytest.raises(SystemExit) as salida:
        main.main(["--entrada", str(carpeta), "--salida", str(tmp_path)])
    assert salida.value.code == 2 and foto.read_bytes() == original
    assert not (tmp_path / "resultados.csv").exists()
