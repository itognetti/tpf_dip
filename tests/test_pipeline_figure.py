from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.pipeline_figure import graficar_recorrido, recorrer_pipeline
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
