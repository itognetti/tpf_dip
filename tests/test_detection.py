from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree

import cv2
import numpy as np
import pandas as pd
import pytest

from scripts.run_detection import casos_localizacion, elegir_preprocesamiento, probar_localizacion
from src import detection as det
from src import evaluation as ev
from src.ocr import ResultadoOCR

DATOS = Path(__file__).resolve().parents[1] / "data"


def test_sobel_con_signo_rango_y_orientacion():
    imagen = np.zeros((25, 45), dtype=np.uint8)
    imagen[:, 12:30] = 255
    gx, gy, magnitud = det.sobel(imagen)
    assert gx.dtype == np.float32 and gx.min() < 0 < gx.max()
    assert not gy.any() and np.array_equal(magnitud, np.abs(gx))
    assert det.intensidad_gradiente(gx).max() == 255
    assert not det.intensidad_gradiente(gy).any()
    assert cv2.countNonZero(det.canny(imagen)) == 50


@pytest.mark.parametrize("operacion", det.OPERACIONES)
def test_morfologia_rectangular(operacion):
    mascara = np.zeros((25, 45), np.uint8)
    mascara[9:16, 10:15] = 255
    mascara[9:16, 17:22] = 255
    copia = mascara.copy()
    salida = det.morfologia_rectangular(mascara, operacion, (5, 3))
    assert np.array_equal(copia, mascara)
    assert salida.shape == mascara.shape and set(np.unique(salida)) <= {0, 255}
    if operacion in {"cierre", "dilatacion"}:
        assert salida[12, 15] == 255
    else:
        assert salida[12, 15] == 0


def test_coordenadas_usan_escalas_reales_y_extremos_semiabiertos():
    entrada = np.zeros((201, 399, 3), np.uint8)
    gris = det.preparar_gris(entrada, 100)
    assert gris.shape == (50, 100)
    assert det.caja_original((10, 20, 25, 40), gris.shape, entrada.shape) == (39, 80, 100, 161)
    assert det.caja_original((-1, -1, 105, 51), gris.shape, entrada.shape) == (0, 0, 399, 201)


def test_contornos_rectangularidad_y_seleccion():
    mascara = np.zeros((180, 430), np.uint8)
    cv2.rectangle(mascara, (90, 70), (210, 105), 255, -1)
    cv2.rectangle(mascara, (14, 14), (25, 20), 255, -1)
    bordes = mascara.copy()
    bordes[:, ::4] = 0
    bordes[:, 1::4] = 0
    candidatos = det.extraer_candidatos(mascara, bordes, mascara.shape, det.ConfigDetector())
    bueno = det.seleccionar_candidato(candidatos)
    assert bueno.caja == (90, 70, 211, 106)
    assert bueno.superficie == 4200 and bueno.rectangularidad == 1
    assert bueno.rectangulo_rotado.shape == bueno.poligono.shape == (4, 2)
    malo = next(c for c in candidatos if c.rechazos)
    assert "dimensiones" in malo.rechazos
    assert det.seleccionar_candidato([replace(malo, puntaje=1), bueno]) is bueno
    assert det.rectangularidad(10, 5, 4) == 0.5
    assert det.rectangularidad(0, 10, 0) == 0


def test_detecta_sin_acceso_a_xml_y_sin_modificar_foto(monkeypatch):
    casos, ambito = casos_localizacion(DATOS, sintetico=True)
    caso = casos[0]
    foto = caso["sintetica"]
    copia = foto.copy()

    def sin_xml(*args, **kwargs):
        raise AssertionError("El detector consultó un XML")

    monkeypatch.setattr(ElementTree, "parse", sin_xml)
    detalle, _ = probar_localizacion(DATOS, casos, det.METODOS_DETECCION, det.ConfigDetector(), ambito)
    assert len(detalle) == 8
    assert detalle.loc[detalle.metodo.eq("D1"), "iou"].min() >= 0.75
    assert np.array_equal(copia, foto)


def test_foto_vacia_de_bordes_y_configuracion_invalida():
    resultado = det.detectar_patente(np.zeros((60, 100), np.uint8), "D3")
    assert resultado.seleccionado is None and resultado.motivo == "sin_contornos"
    with pytest.raises(ValueError, match="uint8"):
        det.detectar_patente(np.zeros((50, 90), np.float32))
    with pytest.raises(ValueError):
        det.ConfigDetector(kernel_canny=(3,))


def test_orden_y_rectificacion_de_un_trapecio():
    vertices = np.float32([[12, 15], [103, 25], [96, 64], [7, 58]])
    desorden = vertices[[2, 0, 3, 1]]
    assert np.array_equal(det.ordenar_vertices(desorden), vertices)
    foto = np.zeros((90, 135, 3), np.uint8)
    cv2.fillConvexPoly(foto, vertices.astype(np.int32), (200, 160, 100))
    copia = foto.copy()
    recta = det.rectificar_perspectiva(foto, desorden)
    assert 2 <= recta.shape[1] / recta.shape[0] <= 2.5
    assert list(recta[recta.shape[0] // 2, 20]) == [200, 160, 100]
    assert np.array_equal(foto, copia)
    with pytest.raises(ValueError, match="fuera"):
        det.rectificar_perspectiva(foto, vertices + [90, 0])
    with pytest.raises(ValueError, match="degenerado"):
        det.ordenar_vertices(np.zeros((4, 2)))


def test_recorte_en_limite_y_alternativa_sin_cuadrilatero():
    foto = np.zeros((65, 120, 3), np.uint8)
    recorte = det.recortar_rectangulo(foto, (5, 4, 120, 65))
    assert recorte.shape == (61, 115, 3) and not np.shares_memory(foto, recorte)
    with pytest.raises(ValueError):
        det.recortar_rectangulo(foto, (5, 4, 121, 65))
    casos, _ = casos_localizacion(DATOS, sintetico=True)
    foto = casos[0]["sintetica"]
    candidato = det.detectar_patente(foto).seleccionado
    resultado = det.extraer_patente(foto, replace(candidato, poligono=candidato.poligono[:3]))
    assert resultado.rectificada is None
    assert resultado.motivo == "sin_cuadrilatero_convexo" and resultado.rectangular.size


@pytest.mark.parametrize(("otra", "esperado"), [((0, 0, 20, 20), 1), ((20, 0, 40, 20), 0),
                                                ((10, 0, 30, 20), 1 / 3), (None, 0)])
def test_iou_y_resumen(otra, esperado):
    assert ev.iou((0, 0, 20, 20), otra) == pytest.approx(esperado)
    detalle = pd.DataFrame({
        "ambito": ["fixture"] * 2, "metodo": ["D1"] * 2, "estado": ["detectada", "sin_deteccion"],
        "correcta_050": [True, False], "correcta_075": [True, False],
        "iou": [1.0, 0.0], "ms_deteccion": [2.0, 4.0],
    })
    fila = ev.resumir_localizacion(detalle).iloc[0]
    assert fila.detecciones_fallidas == 1
    assert fila.tasa_iou_050 == fila.iou_medio == 0.5 and fila.iou_con_deteccion == 1


def test_ocr_geometria_misma_configuracion_y_fallo_en_denominador(monkeypatch):
    casos, _ = casos_localizacion(DATOS, sintetico=True)
    foto = casos[0]["sintetica"]
    candidato = det.detectar_patente(foto).seleccionado
    recorte = det.recortar_rectangulo(foto, casos[0]["caja_ref"])
    psm_usados = []

    def lectura(imagen, psm):
        psm_usados.append(psm)
        return ResultadoOCR(psm, "ABC123\n", "ABC123", 93.0, True)

    monkeypatch.setattr(ev, "reconocer_patente", lectura)
    exito = ev.comparar_ocr_localizacion(recorte, foto, candidato, esperado="ABC123", metodo_espacial="M1")
    fallo = ev.comparar_ocr_localizacion(recorte, foto, None, esperado="ABC123", metodo_espacial="M1")
    assert psm_usados == [7, 7, 7]
    assert {f["metodo"] for f in exito + fallo} == {"M1"}
    assert exito[1]["texto_literal"] == "ABC123\n"
    assert fallo[1]["estado"] == "sin_deteccion" and fallo[1]["cer"] == 1
    resumen = ev.resumir_ocr_localizacion(pd.DataFrame(exito + fallo)).set_index("rama")
    assert resumen.loc["referencia", "accuracy"] == 1
    assert resumen.loc["automatica", "accuracy"] == resumen.loc["automatica", "cer"] == 0.5


def test_elegir_preprocesamiento_ignora_etiquetas_y_particiones_ajenas(tmp_path):
    casos = [{"id_imagen": "dev", "patente": "ABC123"}]
    ruta = tmp_path / "experimentos.csv"
    filas = [
        {"ambito": "real_desarrollo", "psm": 7, "metodo": m, "id_imagen": "dev",
         "esperado": "ABC123", "exacta": m == "M6", "levenshtein": int(m != "M6")}
        for m in ev.METODOS
    ]
    filas.append(filas[0] | {"ambito": "evaluacion_final", "metodo": "M0", "exacta": True})
    pd.DataFrame(filas).to_csv(ruta, index=False)
    assert elegir_preprocesamiento(ruta, casos) == ("M6", "etapa3_desarrollo")
    assert elegir_preprocesamiento(ruta, [{"id_imagen": "dev", "patente": "ABC124"}])[0] == "M0"


def _candidato(caja, puntaje):
    vacio = np.zeros((4, 2), dtype=np.float32)
    return det.Candidato(caja, vacio, vacio, vacio, 1.0, 3.0, .01, .8, .2, puntaje, ())


def test_evidencia_de_caracteres_distingue_patente_de_fondo():
    from src.ocr import cargar_imagen
    from src.segmentation import evidencia_caracteres

    datos = Path(__file__).resolve().parents[1] / "data/examples"
    for nombre, cantidad in (("demo_antigua_ABC123", 6), ("demo_mercosur_AB123CD", 7)):
        puntaje, n = evidencia_caracteres(np.asarray(cargar_imagen(datos / f"{nombre}.png")))
        assert n == cantidad and puntaje > .9
    assert evidencia_caracteres(np.full((60, 200, 3), 128, np.uint8)) == (0.0, 0)


def test_reordenamiento_por_caracteres_prefiere_la_patente():
    from src.ocr import cargar_imagen

    placa = np.asarray(cargar_imagen(Path(__file__).resolve().parents[1] / "data/examples/demo_antigua_ABC123.png"))
    h, w = placa.shape[:2]
    foto = np.full((h + 100, 2 * w + 150, 3), 120, np.uint8)
    foto[50:50 + h, 50:50 + w] = placa
    señuelo = _candidato((w + 100, 50, 2 * w + 100, 50 + h), .80)
    patente = _candidato((50, 50, 50 + w, 50 + h), .75)
    candidatos = [señuelo, patente]
    assert det.seleccionar_candidato(candidatos) is señuelo
    assert det.seleccionar_candidato(candidatos, foto, det.ConfigDetector()) is señuelo
    config = det.ConfigDetector(reordenar_k=3, peso_caracteres=.5)
    assert det.seleccionar_candidato(candidatos, foto, config) is patente
    assert patente.evidencia > .9 and señuelo.evidencia == 0.0
    with pytest.raises(ValueError):
        det.ConfigDetector(reordenar_k=0)
