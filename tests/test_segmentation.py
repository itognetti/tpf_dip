import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent
from PIL import Image

from scripts.label_characters import _revisar, caja_seleccion, registrar
from scripts.run_segmentation import evaluar_recorte, resumir_cantidad, resumir_espacial
from src.character_annotations import (
    emparejar_cajas, estado_anotacion, evaluar_espacial, guardar_anotaciones, leer_anotaciones, referencias_xml,
)
from src.segmentation import (
    METODOS, ConfigSegmentacion, binarizar_caracteres, cajas_componentes, cajas_contornos,
    cajas_proyeccion, filtrar_candidatos, normalizar_caracter, quitar_lineas_perifericas,
    segmentar, separar_unidos, unir_fragmentos,
)


@pytest.fixture
def letras():
    binaria = np.zeros((40, 110), dtype=np.uint8)
    for a, b in ((12, 15), (27, 35), (47, 55), (70, 82)):
        binaria[9:33, a:b] = 255
    return binaria


@pytest.mark.parametrize("metodo", METODOS)
def test_binarizacion_polaridad_y_orden_sin_cantidad_esperada(letras, metodo):
    config = ConfigSegmentacion(altura=40, clahe=0, quitar_marcos=False)
    blanca = np.repeat(letras[:, :, None], 3, axis=2)
    copia = blanca.copy()
    directa = segmentar(blanca, metodo, "completa", config)
    inversa = segmentar(255 - blanca, metodo, "completa", config)
    assert len(directa.cajas) == 4 and directa.cajas == inversa.cajas
    assert directa.clara and not inversa.clara
    assert [c[0] for c in directa.cajas] == sorted(c[0] for c in directa.cajas)
    assert set(np.unique(directa.binaria)) <= {0, 255}
    assert np.array_equal(copia, blanca)


@pytest.mark.parametrize("extraer", (cajas_contornos, cajas_componentes, cajas_proyeccion))
def test_extractor_cajas_y_filtro_no_descarta_i(extraer, letras):
    letras[2:4, 2:5] = 255
    cajas = filtrar_candidatos(letras, extraer(letras), ConfigSegmentacion())
    assert len(cajas) == 4 and cajas[0] == (12, 9, 15, 33)


def test_filtro_de_linea_periferica_preserva_trazos(letras):
    copia = letras.copy()
    letras[1, 5:104] = 255
    assert np.array_equal(quitar_lineas_perifericas(letras), copia)
    assert letras[1, 50] == 255


def test_union_de_fragmentos_y_separacion_por_valle():
    unidos, n = unir_fragmentos([(10, 5, 18, 18), (12, 21, 18, 37), (29, 6, 35, 40)], 50)
    assert n == 1 and (10, 5, 18, 37) in unidos
    assert unir_fragmentos([(10, 5, 11, 30), (13, 5, 14, 30)], 40)[1] == 0
    imagen = np.zeros((40, 80), dtype=np.uint8)
    imagen[7:31, 15:34] = 255
    imagen[7:31, 39:53] = 255
    imagen[19, 34:39] = 255
    assert len(cajas_componentes(imagen)) == 1
    separadas = separar_unidos(imagen, (15, 7, 53, 31))
    assert len(separadas) == 2 and all(c[3] - c[1] == 24 for c in separadas)


def test_coordenadas_de_banda_y_redimensionado():
    entrada = np.zeros((50, 120), np.uint8)
    entrada[18:40, 35:42] = 255
    entrada[18:40, 70:77] = 255
    cfg = ConfigSegmentacion(altura=100, clahe=0, quitar_marcos=False)
    s = segmentar(entrada, "S0", "fila", cfg)
    assert s.banda[1] > 0 and len(s.cajas) == 2
    assert abs(s.cajas[0][0] - 35) <= 1 and abs(s.cajas[0][1] - 18) <= 1
    assert abs(s.cajas[0][3] - 40) <= 1


def test_vacios_tipos_y_fallos_de_segmentacion():
    with pytest.raises(ValueError, match="no vacía"):
        segmentar(np.empty((0, 3), np.uint8))
    with pytest.raises(ValueError, match="no vacía"):
        normalizar_caracter(np.empty((0, 0), np.uint8))
    with pytest.raises(ValueError, match="contraste"):
        normalizar_caracter(np.full((10, 10), 255, np.uint8))
    uniforme = np.full((30, 50), 255, np.uint8)
    mascara, _ = binarizar_caracteres(uniforme, "S1", ConfigSegmentacion())
    assert not mascara.any()
    fila, _ = evaluar_recorte(uniforme, 6, "S1", "fila", ConfigSegmentacion(), {"localizacion": "xml"})
    assert fila["estado"] == "sin_regiones" and fila["diferencia"] == -6
    fallo, _ = evaluar_recorte(None, 7, "S0", "fila", ConfigSegmentacion(), {"localizacion": "sin_localizacion"})
    detalle = pd.DataFrame([fallo | {"tipo_recorte": "automatico", "metodo": "S0", "banda": "fila", "formato": "mercosur"}])
    total = resumir_cantidad(detalle).iloc[0]
    assert total.fallos_totales == total.faltantes == total.sin_localizacion == 1


@pytest.mark.parametrize(("w", "h", "interno"), ((3, 25, 20), (19, 22, 20), (40, 12, 14)))
def test_normalizacion_proporciones_momentos_rango_y_vector(w, h, interno):
    entrada = np.zeros((h + 8, w + 9), np.uint8)
    entrada[4:4+h, 2:2+w] = 255
    normalizada = normalizar_caracter(entrada, interno)
    assert normalizada.ajuste.shape == (h, w)
    assert max(normalizada.escalada.shape) == interno
    esperado = (max(1, round(h * interno / max(w, h))), max(1, round(w * interno / max(w, h))))
    assert normalizada.escalada.shape == esperado
    assert normalizada.imagen.shape == (28, 28) and normalizada.vector.shape == (784,)
    assert normalizada.vector.dtype == np.float32 and np.array_equal(normalizada.imagen.ravel(), normalizada.vector)
    assert 0 <= normalizada.vector.min() <= normalizada.vector.max() <= 1
    assert all(abs(14 - c) <= .5 for c in normalizada.centro_masa)
    assert np.count_nonzero(normalizada.escalada) == np.count_nonzero(normalizada.lienzo)
    assert np.array_equal(normalizar_caracter(255 - entrada, interno).lienzo, normalizada.lienzo)


def test_trazo_delgado_y_piezas_persisten_en_mascara():
    largo = np.full((180, 1), 255, np.uint8)
    n = normalizar_caracter(largo, polaridad="claro")
    assert n.escalada.shape == (20, 1) and np.count_nonzero(n.lienzo) == 20
    a = np.zeros((34, 16), np.uint8)
    a[3:12, 6:9] = a[18:31, 6:9] = 255
    normal = normalizar_caracter(a, polaridad="claro")
    assert len(cajas_componentes(normal.lienzo)) == 2


def test_emparejamiento_maximo_y_unico():
    refs = [(0, 0, 10, 10), (5, 0, 15, 10)]
    detectadas = [(2, 0, 12, 10), (0, 0, 10, 10)]
    assert [(i, j) for i, j, _ in emparejar_cajas(refs, detectadas)] == [(0, 1), (1, 0)]
    union = [(0, 0, 20, 10)]
    a, detalle = evaluar_espacial([(0, 0, 10, 10), (10, 0, 20, 10)], union, (0, 0, 18, 10))
    assert a["tp"] == 1 and a["fn"] == 1 and a["unidos"] == 1 and not a["banda_completa"]
    assert sorted(f["iou"] for f in detalle) == [0, .5]


def test_metricas_f1_e_integridad():
    refs = [(0, 0, 12, 10), (20, 0, 32, 10)]
    halladas = [(0, 0, 4, 10), (7, 0, 11, 10), (20, 0, 29, 10)]
    a, _ = evaluar_espacial(refs, halladas, (0, 0, 32, 10))
    assert a["fragmentados"] == a["incompletos"] == 1 and a["tp"] == 1 and a["fp"] == 2
    resumen = resumir_espacial(pd.DataFrame([a | {"metodo": "S0", "banda": "fila"}])).iloc[0]
    assert resumen.recall == .5 and resumen.precision == pytest.approx(1/3) and resumen.f1 == .4


def test_anotacion_omite_reanuda_y_corrige_en_csv_independiente(tmp_path):
    muestra = {"id_imagen": "dev1", "patente": "AB123CD"}
    archivo = tmp_path / "character_annotations.csv"
    anotaciones = {}
    caja = caja_seleccion((15.7, 9.1), (-1, 0), 20, 12)
    assert caja == (0, 0, 16, 10)
    registrar(anotaciones, muestra, 1, caja)
    registrar(anotaciones, muestra, 2, None)
    guardar_anotaciones(archivo, anotaciones)
    anotaciones = leer_anotaciones(archivo)
    assert estado_anotacion(muestra, anotaciones) == "parcial"
    assert referencias_xml(muestra, anotaciones, 90, 20) is None
    for i in range(1, 8):
        registrar(anotaciones, muestra, i, (i * 10, 1, i * 10 + 6, 12))
    guardar_anotaciones(archivo, anotaciones)
    anotaciones = leer_anotaciones(archivo)
    assert estado_anotacion(muestra, anotaciones) == "completa"
    assert referencias_xml(muestra, anotaciones, 90, 20)[0] == (10, 1, 16, 12)


def test_ventana_guarda_dibujo_y_omision(tmp_path, monkeypatch):
    ruta = tmp_path / "manual.csv"
    anotaciones = {}

    def simular(**kwargs):
        fig = plt.gcf()
        fig.canvas.draw()
        eje = fig.axes[0]
        a = eje.transData.transform((4.5, 3.5))
        b = eje.transData.transform((12.5, 14.5))
        for nombre, punto in (("button_press_event", a), ("motion_notify_event", b), ("button_release_event", b)):
            evento = MouseEvent(nombre, fig.canvas, *punto, button=1)
            fig.canvas.callbacks.process(nombre, evento)
        for tecla in ("x", "escape"):
            evento = KeyEvent("key_press_event", fig.canvas, key=tecla)
            fig.canvas.callbacks.process("key_press_event", evento)

    monkeypatch.setattr(plt, "show", simular)
    muestra = {"id_imagen": "dev", "patente": "ABC123"}
    assert _revisar(Image.new("RGB", (40, 20)), muestra, anotaciones, ruta)
    guardadas = leer_anotaciones(ruta)
    assert guardadas["dev", 1]["caracter"] == "A" and guardadas["dev", 1]["estado"] == "anotado"
    assert guardadas["dev", 2]["caracter"] == "B" and guardadas["dev", 2]["estado"] == "omitido"
