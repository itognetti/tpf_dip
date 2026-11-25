"""Ejecuta y exporta la comparación de la etapa 4 exclusivamente en desarrollo."""

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract
from matplotlib.patches import Rectangle
from PIL import Image

from src.dataset import CAJA
from src.detection import ConfigDetector, METODOS_DETECCION, detectar_patente, extraer_patente
from src.evaluation import (
    REFERENCIAS_SINTETICAS, comparar_ocr_localizacion, graficar_localizacion, iou, muestras_desarrollo,
    resumir_localizacion, resumir_ocr_localizacion,
)
from src.ocr import ImagenInvalidaError, cargar_imagen, configurar_tesseract
from src.preprocessing import METODOS


def casos_localizacion(datos, sintetico=False):
    """Usa cajas de referencia solo en desarrollo o fabrica dos escenas de control."""
    if not sintetico:
        muestras, ambito = muestras_desarrollo(datos)
        if ambito == "real_desarrollo":
            return [fila | {"caja_ref": tuple(int(fila[clave]) for clave in CAJA)} for fila in muestras], ambito

    casos = []
    for indice, ruta in enumerate(REFERENCIAS_SINTETICAS):
        recorte = np.asarray(cargar_imagen(datos / ruta))
        ancho, alto = 190, 48
        placa = cv2.resize(recorte, (ancho, alto), interpolation=cv2.INTER_AREA)
        foto = np.full((600, 900, 3), 58 + 30 * indice, dtype=np.uint8)
        x, y = 350, 385 - 35 * indice
        foto[y:y + alto, x:x + ancho] = placa
        casos.append({"id_imagen": Path(ruta).stem, "caja_ref": (x, y, x + ancho, y + alto), "sintetica": foto})
    return casos, "sintetico_funcional"


def cargar_foto(datos, caso):
    if "sintetica" in caso:
        return caso["sintetica"]
    return np.asarray(cargar_imagen(datos / caso["original"]))


def _dibujar_caja(eje, caja, color, etiqueta, ancho=2, estilo="-"):
    x0, y0, x1, y1 = caja
    eje.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color=color,
                            linewidth=ancho, linestyle=estilo, label=etiqueta))


def mostrar_candidatos(eje, foto, deteccion, referencia=None, *, maximos=4):
    """Muestra los candidatos de mayor puntaje y su estado, en coordenadas originales."""
    eje.imshow(foto, extent=(0, foto.shape[1], foto.shape[0], 0))
    if maximos:
        seleccion = deteccion.seleccionado
        aceptados = [c for c in deteccion.candidatos if not c.rechazos and c is not seleccion]
        rechazados = [c for c in deteccion.candidatos if c.rechazos]
        for grupo, etiqueta, color in (
            (aceptados, "Aceptado", "#4ca9df"), (rechazados, "Rechazado", "#ef6c60"),
        ):
            for indice, candidato in enumerate(grupo[:maximos]):
                rotado = np.vstack([candidato.rectangulo_rotado, candidato.rectangulo_rotado[0]])
                eje.plot(*rotado.T, color=color, linewidth=1.1, label=etiqueta if indice == 0 else None)
        if seleccion is not None:
            contorno = np.vstack([seleccion.contorno, seleccion.contorno[0]])
            eje.plot(*contorno.T, color="#45e4b5", linewidth=1)
    if deteccion.seleccionado is not None:
        _dibujar_caja(eje, deteccion.seleccionado.caja, "#26ddaa", "Seleccionado", ancho=2)
    if referencia is not None:
        _dibujar_caja(eje, referencia, "#ffbc47", "Referencia", estilo="--")
    eje.set_xlim(0, foto.shape[1])
    eje.set_ylim(foto.shape[0], 0)
    eje.axis("off")
    if eje.get_legend_handles_labels()[0]:
        eje.legend(loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8, frameon=False)


def probar_localizacion(datos, casos, metodos, config, ambito):
    """Registra todos los intentos, con tiempo exclusivo del detector."""
    filas, elegidos = [], {}
    for caso in casos:
        identificador, referencia = caso["id_imagen"], caso["caja_ref"]
        try:
            foto = cargar_foto(datos, caso)
        except (OSError, ImagenInvalidaError) as error:
            foto, error_carga = None, str(error)
        for metodo in metodos:
            resultado, seleccionado, ms = None, None, None
            estado = "error_carga" if foto is None else "error_deteccion"
            motivo = error_carga if foto is None else ""
            if foto is not None:
                inicio = perf_counter()
                try:
                    resultado = detectar_patente(foto, metodo, config)
                except (ValueError, cv2.error) as error:
                    motivo = str(error)
                else:
                    seleccionado = resultado.seleccionado
                    estado = "detectada" if seleccionado else "sin_deteccion"
                    motivo = resultado.motivo
                ms = (perf_counter() - inicio) * 1000
            candidatos = resultado.candidatos if resultado else []
            caja = seleccionado.caja if seleccionado else None
            solapamiento = iou(referencia, caja)
            ejemplos = [{
                "caja": c.caja, "puntaje": round(c.puntaje, 4), "aspecto": round(c.aspecto, 4),
                "area_relativa": round(c.area_relativa, 5), "rectangularidad": round(c.rectangularidad, 4),
                "densidad": round(c.densidad, 4), "rechazos": c.rechazos,
            } for c in candidatos[:5]]
            filas.append({
                "ambito": ambito, "id_imagen": identificador, "metodo": metodo,
                "caja_referencia": json.dumps(referencia), "caja_detectada": json.dumps(caja) if caja else "",
                "candidatos": len(candidatos), "aceptados": sum(not c.rechazos for c in candidatos),
                "puntaje": seleccionado.puntaje if seleccionado else None,
                "iou": solapamiento, "correcta_050": solapamiento >= 0.5, "correcta_075": solapamiento >= 0.75,
                "ms_deteccion": ms, "estado": estado, "motivo": motivo,
                "rechazos": json.dumps(Counter(r for c in candidatos for r in c.rechazos)),
                "candidatos_top5": json.dumps(ejemplos),
            })
            elegidos[metodo, identificador] = seleccionado
    return pd.DataFrame(filas), elegidos


def elegir_preprocesamiento(ruta, casos):
    """Usa experimentos reales de PSM 7 si cubren exactamente el desarrollo actual."""
    if ruta.is_file():
        tabla = pd.read_csv(ruta, keep_default_na=False)
        ids = {c["id_imagen"]: c["patente"] for c in casos}
        if {"ambito", "psm", "metodo", "id_imagen", "esperado", "exacta", "levenshtein"} <= set(tabla):
            tabla = tabla[tabla["ambito"].eq("real_desarrollo") & tabla["psm"].eq(7)]
            comparables = []
            for metodo, grupo in tabla.groupby("metodo"):
                if (metodo in METODOS and len(grupo) == len(ids)
                        and dict(zip(grupo["id_imagen"], grupo["esperado"])) == ids):
                    comparables.append((int(grupo["exacta"].eq(True).sum()), -grupo["levenshtein"].sum(), metodo))
            if {fila[2] for fila in comparables} == METODOS.keys():
                mejor = max(comparables, key=lambda x: (x[0], x[1], -int(x[2][1:])))
                return mejor[2], "etapa3_desarrollo"
    return "M0", "sin_resultados_completos_de_etapa3"


def _comparar_ocr(datos, casos, elegidos, detector, metodo):
    filas = []
    for caso in casos:
        try:
            referencia = np.asarray(cargar_imagen(datos / caso["imagen"]))
        except (OSError, ImagenInvalidaError):
            referencia = None
        seleccionado = elegidos[detector, caso["id_imagen"]]
        try:
            foto = cargar_foto(datos, caso) if seleccionado is not None else None
        except (OSError, ImagenInvalidaError):
            foto = None
        registros = comparar_ocr_localizacion(
            referencia, foto, seleccionado, esperado=caso["patente"], metodo_espacial=metodo
        )
        for fila in registros:
            filas.append(fila | {"ambito": "real_desarrollo", "id_imagen": caso["id_imagen"], "detector": detector})
    return pd.DataFrame(filas)


def _guardar_ejemplo(salida, nombre, metodo, foto, deteccion, referencia, iou_observado, extraida):
    fig, eje = plt.subplots(figsize=(8, 5), layout="constrained")
    mostrar_candidatos(eje, foto, deteccion, referencia, maximos=0)
    fig.suptitle(f"{metodo} · {nombre.replace('_', ' ')} · IoU {iou_observado:.2f}")
    fig.savefig(salida / "detected" / f"detection_{nombre}.png",
                dpi=125, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    if extraida is not None:
        Image.fromarray(extraida.rectangular).save(salida / "processed" / f"detection_{nombre}_rectangular.png")
        if extraida.rectificada is not None:
            Image.fromarray(extraida.rectificada).save(salida / "processed" / f"detection_{nombre}_rectificada.png")


def guardar_ejemplos(datos, casos, detalle, metodo, config, salida, elegidos=None):
    por_id = {c["id_imagen"]: c for c in casos}
    intentos = detalle[detalle["metodo"].eq(metodo)]
    for nombre, seleccion in [
        ("correcta", intentos[intentos["correcta_050"]].nlargest(1, "iou")),
        ("sin_candidato", intentos[intentos["estado"].eq("sin_deteccion")].head(1)),
        ("caja_incorrecta", intentos[intentos["estado"].eq("detectada") & ~intentos["correcta_050"]].nsmallest(1, "iou")),
    ]:
        if seleccion.empty:
            continue
        fila = seleccion.iloc[0]
        caso = por_id[fila["id_imagen"]]
        foto = cargar_foto(datos, caso)
        deteccion = detectar_patente(foto, metodo, config)
        extraida = extraer_patente(foto, deteccion.seleccionado) if deteccion.seleccionado else None
        _guardar_ejemplo(salida, nombre, metodo, foto, deteccion, caso["caja_ref"], fila["iou"], extraida)

    correctas = intentos[intentos["correcta_050"]].sort_values("iou", ascending=False)
    for fila in correctas.to_dict("records"):
        caso = por_id[fila["id_imagen"]]
        if elegidos is not None and elegidos[metodo, caso["id_imagen"]].poligono.shape != (4, 2):
            continue
        foto = cargar_foto(datos, caso)
        deteccion = detectar_patente(foto, metodo, config)
        extraida = extraer_patente(foto, deteccion.seleccionado)
        if extraida.rectificada is not None:
            _guardar_ejemplo(salida, "rectificacion", metodo, foto, deteccion,
                             caso["caja_ref"], fila["iou"], extraida)
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metodos", nargs="+", choices=METODOS_DETECCION, default=METODOS_DETECCION)
    parser.add_argument("--config", type=Path, help="JSON con campos de ConfigDetector o la configuración exportada")
    parser.add_argument("--salida", type=Path, default=Path("results"))
    parser.add_argument("--sintetico", action="store_true")
    parser.add_argument("--sin-ocr", action="store_true")
    parser.add_argument("--preprocesamiento", choices=METODOS)
    parser.add_argument("--resultados-etapa3", type=Path, default=Path("results/stage3_desarrollo/preprocessing_experiments.csv"))
    args = parser.parse_args()
    datos = Path("data")
    salida = args.salida
    try:
        parametros = json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else {}
        if not isinstance(parametros, dict):
            raise ValueError("La configuración debe ser un objeto JSON.")
        config = ConfigDetector(**parametros.get("detector", parametros))
        casos, ambito = casos_localizacion(datos, args.sintetico)
    except (ValueError, TypeError, OSError) as error:
        parser.exit(1, f"{error}\n")

    metodos = tuple(dict.fromkeys(args.metodos))
    detalle, elegidos = probar_localizacion(datos, casos, metodos, config, ambito)
    resumen = resumir_localizacion(detalle)
    detector = resumen.sort_values(
        ["tasa_iou_050", "tasa_iou_075", "iou_medio", "metodo"], ascending=[False, False, False, True]
    ).iloc[0]["metodo"]
    print(resumen.round(4).to_string(index=False), flush=True)
    for carpeta in ("graphs", "detected", "processed"):
        (salida / carpeta).mkdir(parents=True, exist_ok=True)
    detalle.to_csv(salida / "detection_experiments.csv", index=False)
    resumen.to_csv(salida / "detection_summary.csv", index=False)
    fig = graficar_localizacion(resumen)
    fig.savefig(salida / "graphs/detection_comparison.png", dpi=140)
    plt.close(fig)
    guardar_ejemplos(datos, casos, detalle, detector, config, salida, elegidos)

    registro = {"ambito": ambito, "detector": asdict(config), "metodos": metodos, "detector_ocr": detector,
                "opencv": cv2.__version__, "ids": [c["id_imagen"] for c in casos], "ocr": "omitido"}
    if not args.sin_ocr and ambito == "real_desarrollo":
        configurar_tesseract()
        try:
            version = str(pytesseract.get_tesseract_version())
            if "eng" not in pytesseract.get_languages():
                raise ValueError("Tesseract requiere el modelo eng.")
        except (OSError, pytesseract.TesseractError, ValueError) as error:
            print(f"OCR omitido: {error}")
            registro["ocr"] = str(error)
        else:
            metodo, criterio = elegir_preprocesamiento(args.resultados_etapa3, casos)
            if args.preprocesamiento:
                metodo, criterio = args.preprocesamiento, "exploracion_explicita"
            comparacion = _comparar_ocr(datos, casos, elegidos, detector, metodo)
            comparacion.to_csv(salida / "detection_ocr_comparison.csv", index=False)
            print(f"OCR · {detector} · {metodo} · PSM 7")
            print(resumir_ocr_localizacion(comparacion).round(4).to_string(index=False))
            registro["ocr"] = {"psm": 7, "idioma": "eng", "metodo": metodo, "criterio": criterio, "tesseract": version}
    (salida / "detection_config.json").write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
