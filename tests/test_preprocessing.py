import numpy as np
import pytest

from src import preprocessing as pre


def test_grises_respeta_rgb_y_copia():
    rgb = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
    gris = pre.escala_grises(rgb)
    assert np.array_equal(gris, [[76, 29]])
    assert not np.shares_memory(pre.escala_grises(gris), gris)
    with pytest.raises(ValueError, match="uint8"):
        pre.escala_grises(rgb.astype(float))


@pytest.mark.parametrize(
    "operacion",
    [pre.gaussiano, pre.mediana, pre.bilateral, pre.ecualizar, pre.clahe,
     pre.laplaciano, pre.high_boost, pre.invertir],
)
def test_filtros_mantienen_dimensiones_tipo_y_original(operacion):
    entrada = np.arange(225, dtype=np.uint8).reshape(15, 15)
    copia = entrada.copy()
    salida = operacion(entrada)
    assert salida.shape == entrada.shape and salida.dtype == np.uint8
    assert np.array_equal(entrada, copia)


@pytest.mark.parametrize(
    "operacion",
    [pre.umbral_global, pre.otsu, pre.umbral_adaptativo,
     pre.erosion, pre.dilatacion, pre.apertura, pre.cierre],
)
def test_operaciones_binarias(operacion):
    entrada = np.tile([0, 255] * 12, (16, 1)).astype(np.uint8)
    salida = operacion(entrada)
    assert salida.shape == entrada.shape and salida.dtype == np.uint8
    assert set(np.unique(salida)) <= {0, 255}


def test_ruido_impulsivo_y_morfologia_tienen_el_efecto_esperado():
    impulso = np.zeros((9, 9), dtype=np.uint8)
    impulso[4, 4] = 255
    assert pre.mediana(impulso)[4, 4] == 0
    assert np.count_nonzero(pre.dilatacion(impulso)) == 9
    assert np.count_nonzero(pre.apertura(impulso)) == 0
    con_hueco = pre.invertir(impulso)
    assert pre.cierre(con_hueco)[4, 4] == 255


@pytest.mark.parametrize("metodo", pre.METODOS)
def test_metodos_preservan_entrada_y_registran_polaridad(metodo):
    entrada = np.zeros((40, 80, 3), dtype=np.uint8)
    entrada[5:-5, 12:-12] = 255
    copia = entrada.copy()
    salida, invertida = pre.aplicar_metodo(entrada, metodo)
    assert np.array_equal(entrada, copia)
    assert salida.shape == (entrada.shape if metodo == "M0" else entrada.shape[:2])
    assert salida.dtype == np.uint8
    if metodo in {"M6", "M7", "M8", "M9"}:
        assert invertida and salida[0, 0] == 255
    else:
        assert not invertida
