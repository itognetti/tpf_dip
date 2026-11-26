import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.pipeline_figure import figura_foto, graficar_recorrido, recorrer_pipeline
from src.dataset import identificador_imagen
from scripts.stage4b_detection import _variantes
from src.detection import ConfigDetector
from src.ocr import cargar_imagen
from src.ocr_preparation import PreparacionOCR
from src.segmentation import ConfigSegmentacion

DATOS = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def vigente():
    metodo, config, _ = _variantes(ConfigDetector())["B3_FILTRO"]
    return {"metodo": metodo, "detector": config, "geometria": "rectangular", "codigo_ocr": "B0P8",
            "preparacion": PreparacionOCR(), "psm": 8, "segmentacion": ConfigSegmentacion(),
            "metodo_seg": "S0", "banda": "fila", "tesseract": ""}


def _foto_sintetica():
    placa = np.asarray(cargar_imagen(DATOS / "examples/demo_antigua_ABC123.png"))
    foto = np.full((900, 1400, 3), 90, np.uint8)
    h, w = placa.shape[:2]
    foto[400:400 + h, 500:500 + w] = placa
    return foto


def test_recorrido_conserva_pasos_y_normaliza(vigente):
    recorrido = recorrer_pipeline(_foto_sintetica(), vigente, ocr=False)
    assert {"gris", "filtrada", "binaria", "mascara"} <= recorrido.deteccion.imagenes.keys()
    assert recorrido.gradiente.shape == recorrido.deteccion.imagenes["gris"].shape
    assert recorrido.deteccion.seleccionado is not None and recorrido.recorte is not None
    assert recorrido.geometria == "rectangular" and recorrido.lectura is None
    assert recorrido.entrada_ocr.dtype == np.uint8
    assert all(c.lienzo.shape == (28, 28) and c.vector.shape == (784,) for c in recorrido.caracteres)
    fig = graficar_recorrido(recorrido, vigente, (500, 400, 800, 480), "ABC123", "prueba")
    assert len(fig.axes) >= 12
    plt.close(fig)


def test_recorrido_sin_deteccion_grafica_motivo(vigente):
    recorrido = recorrer_pipeline(np.full((300, 500, 3), 120, np.uint8), vigente, ocr=False)
    assert recorrido.deteccion.seleccionado is None and recorrido.recorte is None
    fig = graficar_recorrido(recorrido, vigente, titulo="sin caja")
    assert any("Sin recorte" in t.get_text() for t in fig.texts)
    plt.close(fig)


@pytest.fixture
def entorno(tmp_path, vigente):
    """Configuraciones mínimas de 4B y 5 y una foto nueva en disco."""
    s4b, s5, datos = tmp_path / "stage4b", tmp_path / "stage5", tmp_path / "data"
    for carpeta in (s4b, s5, datos):
        carpeta.mkdir()
    (s4b / "config.json").write_text(json.dumps({
        "detector": {"variantes": {"B3_FILTRO": {"metodo": vigente["metodo"],
                                                 "config": asdict(vigente["detector"]), "escalas": []}}},
        "geometria_seleccionada": "rectangular",
        "ocr": {"seleccion": "B0P8", "configuraciones": {
            "B0P8": {"preparacion": asdict(PreparacionOCR()), "psm": 8}}},
    }), encoding="utf-8")
    (s5 / "config.json").write_text(json.dumps({
        "segmentacion": asdict(ConfigSegmentacion()), "provisional": {"metodo": "S0", "banda": "fila"},
    }), encoding="utf-8")
    foto = tmp_path / "nueva.png"
    Image.fromarray(_foto_sintetica()).save(foto)
    return foto, datos, s4b, s5


def _etiquetas(datos, filas):
    with (datos / "ground_truth.csv").open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, ["id_imagen", "particion"])
        escritor.writeheader()
        escritor.writerows(filas)


def test_figura_foto_nueva_sin_etiqueta(entorno):
    foto, datos, s4b, s5 = entorno
    _etiquetas(datos, [{"id_imagen": "otro", "particion": "desarrollo"}])
    fig, resumen = figura_foto(foto, datos, s4b, s5, ocr=False)
    assert resumen["particion"] == "fuera_del_experimento" and resumen["caja"] is not None
    assert resumen["texto_ocr"] is None and resumen["regiones"] >= 0
    plt.close(fig)


def test_figura_foto_rechaza_evaluacion_final(entorno):
    foto, datos, s4b, s5 = entorno
    _etiquetas(datos, [{"id_imagen": identificador_imagen(Image.open(foto).convert("RGB")),
                        "particion": "evaluacion_final"}])
    with pytest.raises(ValueError, match="evaluación final"):
        figura_foto(foto, datos, s4b, s5, ocr=False)
