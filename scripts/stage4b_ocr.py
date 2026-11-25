"""Auditoría de la entrada real y selección incremental de una configuración OCR."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from hashlib import sha256
from shutil import copyfile
from time import perf_counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image

from src.dataset import CAJA, leer_caja_voc
from src.evaluation import distancia_levenshtein
from src.ocr import CARACTERES, cargar_imagen, configurar_tesseract, reconocer_region
from src.ocr_preparation import PreparacionOCR, caja_texto, preparar_region
from src.preprocessing import aplicar_metodo, polaridad_ocr


def _auditar(datos, muestras, previo, salida):
    """Verifica caja, píxeles, etiqueta e insumo M2 sin consultar otras particiones."""
    entradas, filas = {}, []
    for f in muestras:
        ident = f["id_imagen"]
        original = cargar_imagen(datos / f["original"])
        exportada = datos / "raw/roboflow_argentina" / f["ruta_fuente"]
        caja = leer_caja_voc(exportada.with_suffix(".xml"), exportada.name, original.size, f["coordenadas_fuente"])
        recorte = cargar_imagen(datos / f["imagen"])
        entrada = np.asarray(recorte)
        if (caja != tuple(int(f[k]) for k in CAJA) or
                not np.array_equal(original.crop(caja), entrada) or previo.loc[ident, "esperado"] != f["patente"]):
            raise ValueError(f"Desajuste de XML, recorte o etiqueta en {ident}.")
        m2, _ = aplicar_metodo(entrada, "M2")
        preparada, _ = preparar_region(entrada, PreparacionOCR())
        if not np.array_equal(m2, preparada):
            raise ValueError("La entrada del baseline cambió.")
        entradas[ident] = entrada
        _, inv = polaridad_ocr(m2)
        _, y0, _, y1 = caja_texto(m2, "fila")
        filas.append({
            "id_imagen": ident, "formato": f["formato"], "ancho": recorte.width,
            "alto": recorte.height, "aspecto": recorte.width / recorte.height,
            "contraste_p95_p05": np.diff(np.percentile(m2, [5, 95]))[0],
            "var_laplaciano": cv2.Laplacian(m2, cv2.CV_64F).var(), "borde_oscuro": inv,
            "min_intensidad": int(m2.min()), "max_intensidad": int(m2.max()),
            "alto_banda_gradiente_px": y1 - y0 if y1 - y0 < recorte.height else None,
            "estado": previo.loc[ident, "estado"], "exacta": previo.loc[ident, "exacta"],
        })
    auditoria = pd.DataFrame(filas).set_index("id_imagen")
    auditoria.to_csv(salida / "input_audit.csv")
    return entradas, auditoria


def medir_ocr(imagen, esperado, config, psm):
    """Mide un intento, con el mismo denominador para cualquier fallo."""
    fila = {"esperado": esperado, "estado": "sin_deteccion", "texto_literal": "",
            "texto_normalizado": "", "confianza": None, "formato_valido": False,
            "ms_preparacion": 0.0, "ms_ocr": 0.0, "error": ""}
    if imagen is not None:
        inicio = perf_counter()
        try:
            matriz, opciones = preparar_region(imagen, config)
        except (ValueError, cv2.error) as error:
            fila.update(estado="error_preparacion", error=str(error), ms_preparacion=(perf_counter()-inicio)*1000)
        else:
            fila.update(opciones, ancho_ocr=matriz.shape[1], alto_ocr=matriz.shape[0],
                        hash_pixeles=sha256(matriz.tobytes()).hexdigest(), ms_preparacion=(perf_counter()-inicio)*1000)
            inicio = perf_counter()
            try:
                lectura = reconocer_region(Image.fromarray(matriz), psm)
            except (OSError, RuntimeError, pytesseract.TesseractError) as error:
                fila.update(estado="error_ocr", error=str(error))
            else:
                fila.update(estado="ok" if lectura.texto_normalizado else "sin_texto",
                            texto_literal=lectura.texto_original, texto_normalizado=lectura.texto_normalizado,
                            confianza=lectura.confianza, formato_valido=lectura.formato_valido)
            fila["ms_ocr"] = (perf_counter() - inicio) * 1000
    texto = fila["texto_normalizado"]
    fila.update(exacta=texto == esperado, levenshtein=distancia_levenshtein(esperado, texto))
    return fila


def metricas_ocr(tabla):
    """Agrega CER micro, tiempo, formato y confianza auxiliar."""
    return {
        "imagenes": len(tabla), "aciertos": int(tabla.exacta.sum()), "accuracy": tabla.exacta.mean(),
        "cer": tabla.levenshtein.sum() / tabla.esperado.str.len().sum(),
        "sin_texto": int(tabla.estado.eq("sin_texto").sum()),
        "sin_deteccion": int(tabla.estado.eq("sin_deteccion").sum()),
        "errores": int(tabla.estado.str.startswith("error_").sum()),
        "validez": tabla.formato_valido.mean(), "confianza_media": tabla.confianza.mean(),
        "con_confianza": int(tabla.confianza.count()),
        "ms_preparacion": tabla.ms_preparacion.mean(), "ms_ocr": tabla.ms_ocr.mean(),
    }


def _resumir(filas):
    combinadas = pd.DataFrame(filas)
    resumen = []
    for (fase, ident, psm), grupo in combinadas.groupby(["fase", "config", "psm"], sort=False):
        for formato, tabla in [("total", grupo), *grupo.groupby("formato")]:
            resumen.append({"fase": fase, "config": ident, "psm": psm, "formato": formato} | metricas_ocr(tabla))
    return combinadas, pd.DataFrame(resumen)


def _mejores(resumen, ids):
    return (resumen[resumen.formato.eq("total") & resumen.config.isin(ids)]
            .sort_values(["accuracy", "cer"], ascending=[False, True], kind="stable").config.tolist())


def _finalistas_escala(resumen, configuraciones, psm):
    """Combina una preparación y un tamaño finalistas en solo dos ensayos."""
    espacial = _mejores(resumen, ("GRIS", "CLAHE", "CENTRO", "BORDE", "OTSU",
                                 "ADAPT", "BOOST", "MORF", "MARCO", "FILA"))[0]
    escala = _mejores(resumen, ("E2", "E3", "E4", "E3P", "L3P", "H96P"))[0]
    tamano = configuraciones[escala][0]
    combinada = replace(
        configuraciones[espacial][0], factor=tamano.factor, altura=tamano.altura,
        padding=tamano.padding, interpolacion=tamano.interpolacion,
    )
    variantes = {"F_AJUSTE": (combinada, psm), "F_AJUSTE_GRIS": (replace(combinada, filtrado="M1"), psm)}
    vistas = set(configuraciones.values())
    nuevas = {}
    for nombre, par in variantes.items():
        if par not in vistas:
            nuevas[nombre] = par
            vistas.add(par)
    return nuevas


def _muestras_visual(auditoria):
    vacias = auditoria[auditoria.estado.eq("sin_texto")]
    aciertos = auditoria[auditoria.exacta]
    errores = auditoria[~auditoria.exacta & auditoria.estado.eq("ok")]
    return {
        "acierto": aciertos.index[0],
        "incorrecto": errores.index[0],
        "vacio_antiguo": vacias[vacias.formato.eq("antiguo")].index[0],
        "vacio_mercosur": vacias[vacias.formato.eq("mercosur")].index[0],
        "baja_resolucion": vacias.alto.idxmin(),
        "marcos_y_tornillos": aciertos[aciertos.index != aciertos.index[0]].index[0],
    }


def _figuras(salida, entradas, auditoria, detalle, resumen, configuraciones, ganadora):
    casos = _muestras_visual(auditoria)
    vista = detalle[detalle.config.eq(ganadora)].set_index("id_imagen")
    base = detalle[detalle.config.eq("B0")].set_index("id_imagen")
    ejemplos, graficos = salida / "examples", salida / "graphs"
    fig, ejes = plt.subplots(len(casos), 3, figsize=(11, 10.5), layout="constrained")
    for (tipo, ident), (orig, previa, nueva) in zip(casos.items(), ejes):
        orig.imshow(entradas[ident])
        orig.set_title(f"{tipo.replace('_',' ')} · {auditoria.loc[ident,'formato']}", fontsize=9)
        for ax, tabla, config in ((previa, base, "B0"), (nueva, vista, ganadora)):
            preparacion, psm = configuraciones[config]
            imagen, _ = preparar_region(entradas[ident], preparacion)
            destino = ejemplos / f"ocr_{config}_{tipo}.png"
            with pytesseract.pytesseract.save(Image.fromarray(imagen)) as (_, temporal):
                copyfile(temporal, destino)
            with Image.open(destino) as guardada:
                if sha256(guardada.tobytes()).hexdigest() != tabla.loc[ident, "hash_pixeles"]:
                    raise ValueError(f"Los píxeles de depuración difieren del insumo de {ident}.")
            ax.imshow(imagen, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{config} · PSM {psm} · {tabla.loc[ident,'texto_normalizado'] or '(vacío)'}", fontsize=9)
        for ax in (orig, previa, nueva):
            ax.axis("off")
    fig.suptitle("Original XML y píxeles exactos entregados a Tesseract")
    fig.savefig(ejemplos / "ocr_inputs.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for ax, variable in zip(axes, ("ancho", "alto")):
        for estado, color in (("ok", "#267868"), ("sin_texto", "#b85b3f")):
            ax.hist(auditoria.loc[auditoria.estado.eq(estado), variable], bins=12, alpha=0.65, color=color, label=estado)
        ax.set(xlabel=f"{variable.capitalize()} (px)", ylabel="Recortes")
        ax.legend()
    fig.savefig(graficos / "ocr_dimensiones.png", dpi=140)
    plt.close(fig)

    total = resumen[resumen.formato.eq("total")].set_index("config")
    fig, ejes = plt.subplots(2, 1, figsize=(11, 6.3), layout="constrained", sharex=True)
    for ax, variable, titulo in zip(ejes, ("accuracy", "cer"), ("Coincidencia exacta", "CER micro")):
        ax.bar(total.index, total[variable], color=["#267868" if i == ganadora else "#7aa1b5" for i in total.index])
        ax.set_title(titulo)
        ax.grid(axis="y", alpha=0.23)
    ejes[-1].tick_params(axis="x", labelrotation=45)
    fig.savefig(graficos / "ocr_comparison.png", dpi=140)
    plt.close(fig)


def experimentar_ocr(datos, muestras, historico, salida, trabajadores):
    configurar_tesseract()
    if "eng" not in pytesseract.get_languages():
        raise RuntimeError("Tesseract requiere eng para reproducir la etapa 4B.")
    previo = pd.read_csv(historico / "detection_ocr_comparison.csv", keep_default_na=False)
    previo = previo[previo.rama.eq("referencia")].set_index("id_imagen")
    entradas, auditoria = _auditar(datos, muestras, previo, salida)
    base = PreparacionOCR()
    filas, configuraciones = [], {}

    with ThreadPoolExecutor(max_workers=trabajadores) as pool:
        def ejecutar(fase, variantes):
            for nombre, (config, psm) in variantes.items():
                configuraciones[nombre] = config, psm
                intentos = pool.map(lambda f: medir_ocr(entradas[f["id_imagen"]], f["patente"], config, psm), muestras)
                filas.extend(fila | {"id_imagen": f["id_imagen"], "formato": f["formato"],
                                    "fase": fase, "config": nombre, "psm": psm}
                             for f, fila in zip(muestras, intentos))
            _, resumen = _resumir(filas)
            print(f"OCR {fase}: {_mejores(resumen, variantes)}", flush=True)
            return resumen

        escala = {
            "B0": (base, 7), "E2": (replace(base, factor=2), 7),
            "E3": (replace(base, factor=3), 7), "E4": (replace(base, factor=4), 7),
            "P": (replace(base, padding=12), 7), "E3P": (replace(base, factor=3, padding=12), 7),
            "L3P": (replace(base, factor=3, padding=12, interpolacion="lineal"), 7),
            "H96P": (replace(base, altura=96, padding=12), 7),
        }
        resumen = ejecutar("1_escala_margen", escala)
        base_medida = {f["id_imagen"]: f for f in filas if f["config"] == "B0"}
        if any(base_medida[i]["texto_literal"] != previo.loc[i, "texto_literal"] for i in entradas):
            raise ValueError("Los TXT del baseline difieren de la etapa 4.")
        ganador_escala = _mejores(resumen, escala)[0]
        espacial = configuraciones[ganador_escala][0]
        alternativas = {
            "GRIS": replace(espacial, filtrado="M1"), "CLAHE": replace(espacial, filtrado="M5"),
            "CENTRO": replace(espacial, polaridad="central"), "BORDE": replace(espacial, polaridad="borde"),
            "OTSU": replace(espacial, polaridad="central", filtrado="otsu"),
            "ADAPT": replace(espacial, polaridad="central", filtrado="adaptativo"),
            "BOOST": replace(espacial, polaridad="central", filtrado="high_boost"),
            "MORF": replace(espacial, polaridad="central", filtrado="morfologia"),
            "MARCO": replace(espacial, polaridad="central", recorte="marco"),
            "FILA": replace(espacial, polaridad="central", recorte="fila"),
        }
        resumen = ejecutar("2_preparacion", {k: (v, 7) for k, v in alternativas.items()})
        ranking = _mejores(resumen, [*escala, *alternativas])
        preseleccion = configuraciones[ranking[0]][0]
        modos = {f"PSM{p}": (preseleccion, p) for p in (6, 8, 13)}
        if preseleccion != base:
            modos.update({f"B0P{p}": (base, p) for p in (6, 8, 13)})
        resumen = ejecutar("3_psm", modos)
        psm_mejor = configuraciones[_mejores(resumen, [ranking[0], *modos])[0]][1]
        # Prueba dos preparaciones finalistas con un único PSM alternativo.
        alternativo = psm_mejor if psm_mejor != 7 else configuraciones[_mejores(resumen, modos)[0]][1]
        vistos = set(configuraciones.values())
        finalistas = {}
        for ident in ranking:
            prueba = configuraciones[ident][0], alternativo
            if prueba not in vistos:
                finalistas[f"F_{ident}"] = prueba
                vistos.add(prueba)
            if len(finalistas) == 2:
                break
        if finalistas:
            resumen = ejecutar("4_finalistas", finalistas)
        combinadas = _finalistas_escala(resumen, configuraciones, psm_mejor)
        if combinadas:
            ejecutar("4_finalistas", combinadas)

    exportar_ocr(salida, entradas, auditoria, filas, configuraciones, trabajadores)


def exportar_ocr(salida, entradas, auditoria, filas, configuraciones, trabajadores):
    """Exporta las configuraciones comparadas y la selección global."""
    detalle, resumen = _resumir(filas)
    ganadora = _mejores(resumen, configuraciones)[0]
    detalle.to_csv(salida / "ocr_experiments.csv", index=False)
    resumen.to_csv(salida / "ocr_summary.csv", index=False)
    _figuras(salida, entradas, auditoria, detalle, resumen, configuraciones, ganadora)
    registro = {
        "seleccion": ganadora, "ids": list(entradas), "idioma": "eng", "whitelist": CARACTERES,
        "tesseract": str(pytesseract.get_tesseract_version()), "trabajadores": trabajadores,
        "omp_thread_limit": os.environ.get("OMP_THREAD_LIMIT"),
        "configuraciones": {i: {"preparacion": asdict(c), "psm": p} for i, (c, p) in configuraciones.items()},
    }
    (salida / "ocr_config.json").write_text(json.dumps(registro, indent=2), encoding="utf-8")
    print(resumen[resumen.formato.eq("total")].round(4).to_string(index=False), flush=True)
