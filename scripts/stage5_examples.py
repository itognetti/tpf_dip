"""Figuras de segmentación y normalización a partir de ensayos reales."""

from dataclasses import replace
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from src.segmentation import normalizar_caracter, segmentar


def _marcar(eje, caja, color="#55edaf", indice=None):
    x0, y0, x1, y1 = caja
    eje.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color=color, linewidth=1.2))
    if indice is not None:
        eje.text(x0, max(0, y0 - 1), str(indice), color=color, fontsize=8, va="bottom")


def _imagen(eje, imagen):
    h, w = imagen.shape[:2]
    eje.imshow(imagen, cmap="gray", vmin=0, vmax=255, extent=(0, w, h, 0), interpolation="nearest")
    eje.axis("off")


def _regiones(eje, resultado, sobre_original=None):
    _imagen(eje, resultado.binaria if sobre_original is None else sobre_original)
    cajas = resultado.cajas_trabajo if sobre_original is None else resultado.cajas
    for i, caja in enumerate(cajas, 1):
        _marcar(eje, caja, indice=i)


def comparacion_metodos(imagen, metodos, banda, config, titulo):
    fig, ejes = plt.subplots(len(metodos) + 1, 2, figsize=(10, 2.0 * (len(metodos) + 1)),
                             layout="constrained", squeeze=False, gridspec_kw={"width_ratios": [1.35, 1]})
    inicial = segmentar(imagen, metodos[0], banda, config)
    _imagen(ejes[0, 0], imagen)
    _marcar(ejes[0, 0], inicial.banda, "#ffbc47")
    ejes[0, 0].set_title("Recorte y banda (amarillo)")
    ejes[0, 1].axis("off")
    ejes[0, 1].text(0, .5, f"Banda {banda}\nCajas ordenadas de izquierda a derecha",
                    transform=ejes[0, 1].transAxes, va="center")
    for (izquierda, derecha), metodo in zip(ejes[1:], metodos):
        resultado = segmentar(imagen, metodo, banda, config)
        _regiones(izquierda, resultado)
        izquierda.set_title(f"{metodo} · {len(resultado.cajas)} regiones")
        derecha.fill_between(np.arange(len(resultado.perfil)), resultado.perfil, color="#27756d", alpha=.85)
        if metodo == "S2":
            derecha.axhline(ceil(resultado.binaria.shape[0] * config.proyeccion_min),
                             color="#ae5e45", linestyle="--", label="Umbral S2")
            derecha.legend(loc="upper right", fontsize=8)
        derecha.set(xlabel="Columna", ylabel="Píxeles blancos", xlim=(0, len(resultado.perfil)))
    fig.suptitle(titulo)
    return fig


def normalizacion(ejemplos, metodo, banda, config):
    regiones = []
    for imagen, posicion in ejemplos:
        resultado = segmentar(imagen, metodo, banda, config)
        if not resultado.cajas:
            continue
        if posicion is None:
            aspectos = [(a[2] - a[0]) / (a[3] - a[1]) for a in resultado.cajas_trabajo]
            indices = list(dict.fromkeys((int(np.argmin(aspectos)), int(np.argmax(aspectos)))))
        else:
            indices = [posicion - 1] if posicion <= len(resultado.cajas) else []
        regiones.extend((imagen, resultado, i) for i in indices)
    if not regiones:
        return None
    fig, ejes = plt.subplots(len(regiones), 4, figsize=(10, 2.7 * len(regiones)),
                             layout="constrained", squeeze=False)
    for fila, (imagen, s, i) in zip(ejes, regiones):
        x0, y0, x1, y1 = s.cajas[i]
        a, b, c, d = s.cajas_trabajo[i]
        n = normalizar_caracter(s.binaria[b:d, a:c], config.lado_interno, polaridad="claro")
        pasos = [
            ("Recorte RGB", imagen[y0:y1, x0:x1]), ("Máscara ajustada", n.ajuste),
            (f"Lado máximo {config.lado_interno}", n.escalada), ("28 × 28 · centro de masa", n.lienzo),
        ]
        for eje, (titulo, etapa) in zip(fila, pasos):
            _imagen(eje, etapa)
            eje.set_title(titulo, fontsize=9)
        cx, cy = n.centro_masa
        fila[-1].scatter(cx + .5, cy + .5, s=16, marker="+", color="#ffbc47")
    fig.suptitle("Regiones estrecha y ancha · vector float32 de 784 componentes")
    return fig


def _guardar(fig, ruta):
    fig.savefig(ruta, dpi=135)
    plt.close(fig)


def _reparacion(imagen, metodo, banda, config):
    sin_reparar = segmentar(imagen, metodo, banda, replace(config, unir_fragmentos=False, separar_unidos=False))
    reparada = segmentar(imagen, metodo, banda, config)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3), layout="constrained")
    for eje, seg, titulo in ((a, sin_reparar, "Sin unión o separación"),
                             (b, reparada, f"Uniones {reparada.uniones} · divisiones {reparada.separaciones}")):
        _regiones(eje, seg)
        eje.set_title(titulo)
    fig.suptitle("Reparaciones propuestas antes del filtro final")
    return fig


def generar_figuras(salida, recortes, xml, detalle, resumen, metodos, elegido, config, especial,
                    detector="B3_FILTRO"):
    metodo, banda = elegido
    archivos, graficos = salida / "examples", salida / "graphs"
    ejemplos = {}
    total = resumen[resumen.formato.eq("total")].copy()
    tipos = list(recortes)
    fig, ejes = plt.subplots(1, len(tipos), figsize=(5.2 * len(tipos), 3.9), squeeze=False, layout="constrained")
    for eje, tipo in zip(ejes[0], tipos):
        tabla = total[total.tipo_recorte.eq(tipo)].pivot(index="metodo", columns="banda", values="tasa_cantidad")
        tabla.plot.bar(ax=eje, rot=0, color=["#25756d", "#bd7454", "#6e91a7"][:len(tabla.columns)])
        eje.set(title="Recorte XML" if tipo == "xml" else f"Automático {detector}",
                xlabel="", ylabel="Proporción con cantidad correcta", ylim=(0, 1))
        eje.legend(title="Banda", fontsize=8)
        eje.grid(axis="y", alpha=.25)
    _guardar(fig, graficos / "segmentation_counts.png")

    seleccion = detalle[detalle.tipo_recorte.eq("xml") & detalle.metodo.eq(metodo) & detalle.banda.eq(banda)]
    correctas = seleccion[seleccion.cantidad_correcta]
    if not correctas.empty:
        fila = correctas.iloc[0]
        entrada = xml[fila.id_imagen]
        ejemplos["cantidad_correcta"] = fila.id_imagen
        fig = comparacion_metodos(entrada, metodos, banda, config,
                                  f"Mismo recorte XML · {fila.formato} · {fila.esperados} caracteres esperados")
        _guardar(fig, archivos / "xml_comparacion.png")
        # Selección pedagógica revisada visualmente en desarrollo.
        revisadas = [("7a99c5e24ba5924621b1", 2), ("e7eabde847dd518a92d1", 3)]
        normalizadas = None
        if elegido == ("S0", "fila") and all(ident in xml for ident, _ in revisadas):
            normalizadas = normalizacion([(xml[ident], i) for ident, i in revisadas], metodo, banda, config)
            ejemplos["normalizacion"] = revisadas
        if normalizadas is None:
            normalizadas = normalizacion([(entrada, None)], metodo, banda, config)
        if normalizadas:
            _guardar(normalizadas, archivos / "normalizacion.png")
        bandas = detalle[detalle.tipo_recorte.eq("xml")].banda.unique()
        fig, ejes = plt.subplots(len(bandas), 2, figsize=(9, 1.9 * len(bandas)), layout="constrained", squeeze=False)
        for (a, b), opcion in zip(ejes, bandas):
            resultado = segmentar(entrada, metodo, opcion, config)
            _imagen(a, entrada)
            _marcar(a, resultado.banda, "#ffbc47")
            a.set_title(f"Región {opcion}")
            _regiones(b, resultado)
            b.set_title(f"{metodo} · {len(resultado.cajas)} regiones")
        fig.suptitle("Región completa y variantes de banda del mismo recorte XML")
        _guardar(fig, archivos / "bandas.png")

    fallidas = seleccion[(seleccion.detectados.gt(0)) & ~seleccion.cantidad_correcta].sort_values("diferencia", ascending=False)
    if not fallidas.empty:
        fila = fallidas.iloc[0]
        ejemplos["cantidad_incorrecta"] = fila.id_imagen
        fig = comparacion_metodos(xml[fila.id_imagen], metodos, banda, config,
                                  f"Cantidad incorrecta en {metodo} · {fila.esperados} esperados")
        _guardar(fig, archivos / "xml_fallo.png")
    for columna, nombre in (("separaciones", "unidos"), ("uniones", "fragmentados")):
        reparadas = seleccion[seleccion[columna].gt(0)]
        if not reparadas.empty:
            if columna == "uniones":
                reparadas = reparadas.sort_values(["cantidad_correcta", columna], ascending=False)
            else:
                reparadas = reparadas.sort_values(columna, ascending=False)
            fila = reparadas.iloc[0]
            ejemplos[nombre] = fila.id_imagen
            fig = _reparacion(xml[fila.id_imagen], metodo, banda, config)
            _guardar(fig, archivos / f"{nombre}.png")

    auto = recortes.get("automatico", {})
    if especial in auto and especial in xml:
        fig, ejes = plt.subplots(2, 2, figsize=(9, 5), layout="constrained")
        fila = detalle[detalle.id_imagen.eq(especial) & detalle.tipo_recorte.eq("automatico")].iloc[0]
        for (a, b), entrada, titulo in zip(ejes, (xml[especial], auto[especial]),
                                           ("Recorte XML", f"{detector} · IoU {fila.iou_patente:.3f}")):
            s = segmentar(entrada, metodo, banda, config)
            _imagen(a, entrada)
            _marcar(a, s.banda, "#ffbc47")
            a.set_title(titulo)
            _regiones(b, s)
            b.set_title(f"{metodo} · {len(s.cajas)} regiones")
        fig.suptitle("Una caja con IoU aceptable puede cortar trazos")
        _guardar(fig, archivos / "automatico_recortado.png")
        ejemplos["recorte_automatico_historico"] = especial
    return ejemplos
