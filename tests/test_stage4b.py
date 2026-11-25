from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytesseract

from scripts.run_detection import casos_localizacion
from scripts.stage4b_detection import detectar_variante, diagnosticar_candidatos, preferencias_pipeline
from scripts.stage4b_ocr import medir_ocr, metricas_ocr
from src.detection import ConfigDetector, Deteccion, distribucion_vertical, extraer_variante, kernel_relativo
from src.ocr import cargar_imagen, configurar_tesseract, reconocer_patente, reconocer_region
from src.ocr_preparation import PreparacionOCR, caja_texto, preparar_region, reescalar
from src.preprocessing import aplicar_metodo

DATOS = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.parametrize("factor", (1, 2, 3, 4))
def test_escala_preserva_aspecto_y_tipo(factor):
    gris = np.arange(40 * 100, dtype=np.uint8).reshape(40, 100)
    salida = reescalar(gris, PreparacionOCR(factor=factor))
    assert salida.shape == (40 * factor, 100 * factor)
    assert salida.dtype == np.uint8 and not np.shares_memory(gris, salida)


def test_preparacion_compatible_con_baseline_y_padding():
    original = np.asarray(cargar_imagen(DATOS / "examples/demo_antigua_ABC123.png"))
    historica, _ = aplicar_metodo(original, "M2")
    salida, cambios = preparar_region(original, PreparacionOCR())
    assert np.array_equal(historica, salida) and not cambios["polaridad_invertida"]
    nueva, _ = preparar_region(original, PreparacionOCR(altura=96, padding=12, interpolacion="lineal"))
    assert nueva.shape == (120, 408)
    assert (nueva[:12] == 255).all() and (nueva[:, -12:] == 255).all()


def test_recorta_banda_solo_si_hay_concentracion_y_elige_polaridad_por_pixeles():
    gris = np.zeros((60, 180), np.uint8)
    gris[28:46, 18:162:8] = 240
    copia = gris.copy()
    recorte = caja_texto(gris, "fila")
    assert recorte[1] <= 28 < 46 <= recorte[3] and recorte[3] - recorte[1] < 60
    preparada, cambios = preparar_region(gris, PreparacionOCR(filtrado="M1", polaridad="central", recorte="fila"))
    assert cambios["polaridad_invertida"] and preparada[0, 0] == 255
    assert np.array_equal(gris, copia)
    assert caja_texto(np.full_like(gris, 127), "fila") == (0, 0, 180, 60)


def test_una_llamada_preserva_baseline_y_admite_psm_nuevos():
    configurar_tesseract()
    try:
        idiomas = pytesseract.get_languages()
    except pytesseract.TesseractNotFoundError:
        pytest.skip("Tesseract ausente")
    if "eng" not in idiomas:
        pytest.skip("Datos eng ausentes")
    entrada = cargar_imagen(DATOS / "examples/demo_antigua_ABC123.png")
    assert reconocer_region(entrada, 7) == reconocer_patente(entrada, 7)
    for psm in (6, 8, 13):
        assert reconocer_region(entrada, psm).texto_normalizado == "ABC123"


def test_mediciones_incluyen_reconocimiento_fallido():
    resultado = medir_ocr(None, "ABC123", PreparacionOCR(), 7)
    metricas = metricas_ocr(pd.DataFrame([resultado]))
    assert resultado["levenshtein"] == 6
    assert metricas["cer"] == 1 and metricas["sin_deteccion"] == 1 and metricas["accuracy"] == 0


def test_kernel_relativo_y_distribucion_vertical():
    assert kernel_relativo((480, 960), (.02, .01)) == (19, 5)
    distribuida = np.zeros((20, 40), dtype=bool)
    distribuida[:, ::4] = True
    concentrada = np.zeros_like(distribuida)
    concentrada[:, :10] = distribuida[:, :10]
    assert 0 < distribucion_vertical(concentrada) < distribucion_vertical(distribuida) <= 1
    assert distribucion_vertical(np.zeros_like(distribuida)) == 0


def test_diagnostico_separa_generacion_descarte_y_ranking():
    casos, _ = casos_localizacion(DATOS, sintetico=True)
    caso = casos[0]
    real = detectar_variante(caso["sintetica"], ("D1", ConfigDetector(), ())).seleccionado
    ajeno = replace(real, caja=(0, 0, 60, 20))
    descartado = replace(real, rechazos=("aspecto",))
    for candidatos, elegido, causa in (
        ([ajeno], ajeno, "A_no_generada"),
        ([ajeno, descartado], ajeno, "B_descartada"),
        ([ajeno, real], ajeno, "C_no_seleccionada"),
        ([real, ajeno], real, "correcta"),
    ):
        filas, fila = diagnosticar_candidatos(caso["caja_ref"], Deteccion(candidatos, elegido, "", {}, 5))
        assert fila["diagnostico"] == causa and fila["contornos"] == 5
        assert fila["candidatos_pre"] == len(candidatos)
        assert fila["recall_top3"] == (causa in ("correcta", "C_no_seleccionada"))
        assert filas[0]["rango_pre"] == filas[0]["rango_post"] == 1


def test_multiescala_y_rechazo_de_perspectiva_incompleta():
    casos, _ = casos_localizacion(DATOS, sintetico=True)
    caso = casos[0]
    imagen = caso["sintetica"]
    resultado = detectar_variante(imagen, ("D1", ConfigDetector(), (600, 900, 1200)))
    assert {c.escala for c in resultado.candidatos} == {600, 900}
    candidato = resultado.seleccionado
    assert candidato is not None
    rect, modo = extraer_variante(imagen, replace(candidato, poligono=candidato.poligono[:3]), "perspectiva")
    assert modo == "rectangular"
    x0, y0, x1, y1 = candidato.caja
    assert np.array_equal(rect, imagen[y0:y1, x0:x1])


def test_preferencias_del_pipeline_distinguen_accuracy_y_cer():
    resumen = pd.DataFrame([
        {"pipeline": "anterior", "detector": "D3", "accuracy": .35, "cer": .5},
        {"pipeline": "ambos", "detector": "B3", "accuracy": .34, "cer": .4},
        {"pipeline": "referencia", "detector": "XML", "accuracy": .75, "cer": .1},
    ])
    assert preferencias_pipeline(resumen) == {"por_accuracy": "anterior", "por_cer": "ambos"}
