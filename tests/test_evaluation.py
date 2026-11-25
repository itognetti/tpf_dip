import numpy as np
import pytest
import pytesseract
from PIL import Image

from src import evaluation as evaluacion
from src.dataset import COLUMNAS, guardar_csv
from src.ocr import ResultadoOCR, normalizar_texto


@pytest.mark.parametrize(
    ("esperado", "obtenido", "distancia"),
    [("ABC123", "ABC123", 0), ("ABC123", "ABC12", 1), ("ABC123", "A1C1234", 2),
     ("AB123CD", "", 7), ("", "A", 1)],
)
def test_levenshtein_y_cer(esperado, obtenido, distancia):
    assert evaluacion.distancia_levenshtein(esperado, obtenido) == distancia
    if esperado:
        assert evaluacion.cer(esperado, obtenido) == distancia / len(esperado)
    else:
        with pytest.raises(ValueError, match="referencia"):
            evaluacion.cer(esperado, obtenido)


def test_comparacion_conserva_vacio_error_y_literal(tmp_path, monkeypatch):
    Image.new("RGB", (40, 24), "white").save(tmp_path / "recorte.png")
    metodo = iter(evaluacion.METODOS)

    def reconocer(imagen, psm):
        assert psm == 7
        nombre = next(metodo)
        if nombre == "M1":
            raise pytesseract.TesseractError(1, "fallo de prueba")
        if nombre == "M2":
            return ResultadoOCR(psm, "", "", None, False)
        original = " ABC-124\n" if nombre == "M3" else " ABC-123\n"
        return ResultadoOCR(psm, original, normalizar_texto(original), 0.0, True)

    monkeypatch.setattr(evaluacion, "reconocer_patente", reconocer)
    muestras = [
        {"id_imagen": "foto", "imagen": "recorte.png", "patente": "ABC123"},
        {"id_imagen": "ausente", "imagen": "ausente.png", "patente": "AB123CD"},
    ]
    detalle = evaluacion.evaluar_muestras(tmp_path, muestras, "fixture")
    filas = detalle[detalle["id_imagen"].eq("foto")].set_index("metodo")
    assert filas.loc["M0", "texto_literal"] == " ABC-123\n"
    assert filas.loc["M1", "estado"] == "error_ocr" and filas.loc["M1", "cer"] == 1
    assert filas.loc["M2", "estado"] == "sin_texto" and filas.loc["M2", "texto_normalizado"] == ""
    assert filas.loc["M3", "formato_valido"] and not filas.loc["M3", "exacta"]
    assert detalle[detalle["id_imagen"].eq("ausente")]["estado"].eq("error_carga").all()
    resumen = evaluacion.resumir_experimentos(detalle)
    total = resumen[resumen["formato"].eq("total")].set_index("metodo")
    assert total.loc["M0", "accuracy"] == 0.5 and total.loc["M0", "cer"] == 7 / 13
    assert total.loc["M0", "con_confianza"] == 1 and total.loc["M0", "confianza_media"] == 0
    assert total.loc["M2", "sin_texto"] == total.loc["M2", "errores"] == 1
    assert np.isnan(total.loc["M1", "confianza_media"])


def test_solo_muestras_verificadas_de_desarrollo(tmp_path):
    ruta = tmp_path / "ground_truth.csv"
    filas = [
        {"id_imagen": "1", "imagen": "a.png", "patente": "ABC123", "particion": "evaluacion_final", "estado": "verificada"},
        {"id_imagen": "2", "imagen": "b.png", "patente": "ABC124", "particion": "desarrollo", "estado": "pendiente"},
    ]
    guardar_csv(ruta, filas, COLUMNAS)
    muestras, ambito = evaluacion.muestras_desarrollo(tmp_path)
    assert ambito == "sintetico_demostracion" and len(muestras) == 2
    filas[1]["estado"] = "verificada"
    guardar_csv(ruta, filas, COLUMNAS)
    muestras, ambito = evaluacion.muestras_desarrollo(tmp_path)
    assert ambito == "real_desarrollo" and [m["id_imagen"] for m in muestras] == ["2"]
    filas[1]["patente"] = "ABC123"
    guardar_csv(ruta, filas, COLUMNAS)
    with pytest.raises(ValueError, match="cruza particiones"):
        evaluacion.muestras_desarrollo(tmp_path)
