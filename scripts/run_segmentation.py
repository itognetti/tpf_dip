"""Evalúa S0–S2 y exporta caracteres de desarrollo sin etiquetas automáticas."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from scripts.stage4b_detection import detectar_variante
from src.character_annotations import estado_anotacion, evaluar_espacial, leer_anotaciones, referencias_xml
from src.dataset import CAJA
from src.detection import ConfigDetector, extraer_variante, ordenar_vertices
from src.evaluation import iou, muestras_desarrollo
from src.ocr import ImagenInvalidaError, cargar_imagen
from src.segmentation import BANDAS, CLASES, METODOS, ConfigSegmentacion, normalizar_caracter, segmentar

INVENTARIO = ("id_imagen", "tipo_recorte", "metodo", "banda", "indice_izquierda", "caja",
              "imagen_segmentada", "imagen_normalizada", "centro_x", "centro_y")


def _caja_poligono(caja):
    x0, y0, x1, y1 = caja
    return np.float32(((x0, y0), (x1, y0), (x1, y1), (x0, y1)))


def cargar_automatica(datos, muestra, variante, geometria):
    """Aplica el detector congelado. El XML solo se consulta después de seleccionar."""
    try:
        foto = np.asarray(cargar_imagen(datos / muestra["original"]))
        resultado = detectar_variante(foto, variante)
    except (OSError, ValueError, cv2.error) as error:
        return None, {"localizacion": "error_localizacion", "detalle_localizacion": str(error), "iou_patente": 0.}
    candidato = resultado.seleccionado
    if candidato is None:
        return None, {"localizacion": "sin_localizacion", "detalle_localizacion": resultado.motivo, "iou_patente": 0.}
    referencia = tuple(int(muestra[k]) for k in CAJA)
    solape = iou(referencia, candidato.caja)
    meta = {"localizacion": "valida" if solape >= .5 else "iou_bajo",
            "iou_patente": solape, "caja_patente_automatica": json.dumps(candidato.caja)}
    try:
        recorte, efectiva = extraer_variante(foto, candidato, geometria)
    except (ValueError, cv2.error) as error:
        return None, meta | {"localizacion": "error_extraccion", "detalle_localizacion": str(error)}
    poligono = (ordenar_vertices(candidato.rectangulo_rotado) if efectiva == "rotado"
                else _caja_poligono(candidato.caja))
    poligono_ref = _caja_poligono(referencia)
    cruzada, _ = cv2.intersectConvexConvex(poligono_ref, poligono)
    cobertura = min(1., max(0., cruzada / cv2.contourArea(poligono_ref)))
    return recorte, meta | {"geometria": efectiva, "cobertura_caja_xml": cobertura}


def evaluar_recorte(imagen, esperado, metodo, banda, config, localizacion):
    """Registra cualquier intento, incluidos los recortes automáticos ausentes."""
    fila = {
        "esperados": esperado, "detectados": 0, "diferencia": -esperado, "cantidad_correcta": False,
        "estado": "sin_recorte", "cajas": "[]", "banda_caja": "", "polaridad_clara": "",
        "uniones": 0, "separaciones": 0, "ms_segmentacion": 0.,
    } | localizacion
    if imagen is None:
        return fila, None
    inicio = perf_counter()
    try:
        resultado = segmentar(imagen, metodo, banda, config)
    except (ValueError, cv2.error) as error:
        fila.update(estado="error_segmentacion", detalle_segmentacion=str(error))
        resultado = None
    else:
        cantidad = len(resultado.cajas)
        fila.update(
            detectados=cantidad, diferencia=cantidad - esperado, cantidad_correcta=cantidad == esperado,
            estado="con_regiones" if cantidad else "sin_regiones", cajas=json.dumps(resultado.cajas),
            banda_caja=json.dumps(resultado.banda), polaridad_clara=resultado.clara,
            uniones=resultado.uniones, separaciones=resultado.separaciones,
        )
    fila["ms_segmentacion"] = (perf_counter() - inicio) * 1000
    return fila, resultado


def resumir_cantidad(detalle):
    filas = []
    for (tipo, metodo, banda), grupo in detalle.groupby(["tipo_recorte", "metodo", "banda"], sort=False):
        for formato, tabla in [("total", grupo), *grupo.groupby("formato")]:
            variacion = tabla.diferencia
            correctas = tabla.cantidad_correcta
            filas.append({
                "tipo_recorte": tipo, "metodo": metodo, "banda": banda, "formato": formato,
                "imagenes": len(tabla), "cantidad_correcta": int(correctas.sum()), "tasa_cantidad": correctas.mean(),
                "exceso": int(variacion.gt(0).sum()), "faltantes": int(variacion.lt(0).sum()),
                "fallos_totales": int(tabla.detectados.eq(0).sum()),
                "sin_localizacion": int(tabla.localizacion.isin(("sin_localizacion", "error_localizacion", "error_extraccion")).sum()),
                "iou_bajo": int(tabla.localizacion.eq("iou_bajo").sum()),
                "cantidad_correcta_con_localizacion": int((correctas & tabla.localizacion.isin(("xml", "valida"))).sum()),
                "error_absoluto_medio": variacion.abs().mean(), "ms_promedio": tabla.ms_segmentacion.mean(),
            })
    return pd.DataFrame(filas)


def resumir_espacial(detalle):
    filas = []
    for (metodo, banda), grupo in detalle.groupby(["metodo", "banda"], sort=False):
        tp, fp, fn = map(int, grupo[["tp", "fp", "fn"]].sum())
        filas.append({
            "metodo": metodo, "banda": banda, "patentes": len(grupo),
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.,
            "iou_medio_pares": grupo.iou_acumulado.sum() / tp if tp else None,
            "tasa_todos": grupo.todos.mean(), "tasa_banda_completa": grupo.banda_completa.mean(),
            "bandas_evaluadas": int(grupo.banda_completa.count()),
            "incompletos": int(grupo.incompletos.sum()), "unidos": int(grupo.unidos.sum()),
            "fragmentados": int(grupo.fragmentados.sum()),
        })
    return pd.DataFrame(filas)


def preparar_muestra(muestras, recortes, semilla=42):
    """Elige 20 fotos por formato y tercil de tamaño/contraste sin repetir matrículas."""
    filas = []
    for f in muestras:
        im = recortes.get(f["id_imagen"])
        if im is None:
            continue
        gris = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        filas.append({
            "id_imagen": f["id_imagen"], "formato": f["formato"], "imagen": f["imagen"], "patente": f["patente"],
            "ancho": im.shape[1], "alto": im.shape[0], "contraste": np.diff(np.percentile(gris, [5, 95]))[0],
        })
    medidas = pd.DataFrame(filas)
    if medidas.empty:
        return pd.DataFrame(columns=["id_imagen", "formato", "imagen", "ancho", "alto", "contraste",
                                     "dificultad", "estrato"])
    generador = np.random.default_rng(semilla)
    elegidas, vistas = [], set()
    for _, tabla in medidas.groupby("formato"):
        tabla = tabla.copy()
        tabla["dificultad"] = (.55 * tabla["ancho"].rank(pct=True) + .45 * tabla["contraste"].rank(pct=True))
        terciles = np.array_split(tabla.sort_values("dificultad").index, 3)
        bloque = []
        for nombre, corte, cupo in zip(("alta", "media", "baja"), terciles, (4, 3, 3)):
            candidatos = tabla.loc[generador.permutation(corte)].to_dict("records")
            cantidad = 0
            for fila in candidatos:
                if fila["patente"] not in vistas:
                    vistas.add(fila["patente"])
                    bloque.append(fila | {"estrato": nombre})
                    cantidad += 1
                if cantidad == cupo:
                    break
        for fila in tabla.loc[generador.permutation(tabla.index)].to_dict("records"):
            if len(bloque) == 10:
                break
            if fila["patente"] not in vistas:
                vistas.add(fila["patente"])
                bloque.append(fila | {"estrato": "suplemento"})
        elegidas.extend(bloque)
    return pd.DataFrame(elegidas).drop(columns="patente").round(4)


def exportar_caracteres(salida, recortes, tipo, muestras, metodo, banda, config):
    """Guarda las regiones de la configuración provisional con su procedencia."""
    filas = []
    for muestra in muestras:
        ident = muestra["id_imagen"]
        entrada = recortes.get(ident)
        if entrada is None:
            continue
        resultado = segmentar(entrada, metodo, banda, config)
        for indice, (caja, trabajo) in enumerate(zip(resultado.cajas, resultado.cajas_trabajo), 1):
            x0, y0, x1, y1 = trabajo
            mascara = resultado.binaria[y0:y1, x0:x1]
            normalizada = normalizar_caracter(mascara, config.lado_interno, polaridad="claro")
            nombre = f"{tipo}_{ident}_{metodo}_{banda}_{indice:02d}.png"
            original, final = salida / "characters" / nombre, salida / "normalized" / nombre
            Image.fromarray(mascara).save(original)
            Image.fromarray(normalizada.lienzo).save(final)
            filas.append({"id_imagen": ident, "tipo_recorte": tipo, "metodo": metodo, "banda": banda,
                          "indice_izquierda": indice, "caja": json.dumps(caja), "imagen_segmentada": original.as_posix(),
                          "imagen_normalizada": final.as_posix(), "centro_x": normalizada.centro_masa[0],
                          "centro_y": normalizada.centro_masa[1]})
    return filas


def _experimentar(tipo, muestras, recortes, ubicaciones, metodos, bandas, config, anotaciones):
    filas, espaciales, caracteres = [], [], []
    for muestra in muestras:
        ident, palabra = muestra["id_imagen"], muestra["patente"]
        entrada = recortes.get(ident)
        referencias = (referencias_xml(muestra, anotaciones, entrada.shape[1], entrada.shape[0])
                       if entrada is not None and tipo == "xml" else None)
        for metodo in metodos:
            for banda in bandas:
                fila, s = evaluar_recorte(entrada, len(palabra), metodo, banda, config, ubicaciones[ident])
                meta = {"id_imagen": ident, "tipo_recorte": tipo, "formato": muestra["formato"], "metodo": metodo, "banda": banda}
                filas.append(meta | fila)
                if referencias is not None:
                    resultado, pares = evaluar_espacial(referencias, s.cajas if s else [], s.banda if s else None)
                    espaciales.append(meta | resultado)
                    caracteres.extend(meta | p | {"caracter_humano": palabra[p["indice_anotado"] - 1]} for p in pares)
    return filas, espaciales, caracteres


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metodos", nargs="+", choices=METODOS, default=METODOS)
    parser.add_argument("--bandas", nargs="+", choices=BANDAS, default=BANDAS)
    parser.add_argument("--tipo", choices=("xml", "automatico", "ambos"), default="ambos")
    parser.add_argument("--config", type=Path, help="JSON con ConfigSegmentacion o la configuración exportada")
    parser.add_argument("--salida", type=Path, default=Path("results/stage5"))
    args = parser.parse_args()
    salida, datos = args.salida, Path("data")
    if not salida.resolve().is_relative_to(Path("results/stage5").resolve()):
        parser.error("La salida debe estar dentro de results/stage5.")
    plan = json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else {}
    config = ConfigSegmentacion(**plan.get("segmentacion", plan))
    metodos, bandas = tuple(dict.fromkeys(args.metodos)), tuple(dict.fromkeys(args.bandas))
    muestras, ambito = muestras_desarrollo(datos)
    if ambito != "real_desarrollo":
        parser.error("Se requiere una partición de desarrollo con etiquetas verificadas.")
    anotaciones = leer_anotaciones(datos / "character_annotations.csv")
    for nombre in ("examples", "graphs", "characters", "normalized"):
        (salida / nombre).mkdir(parents=True, exist_ok=True)

    xml, meta_xml = {}, {}
    for f in muestras:
        ident = f["id_imagen"]
        try:
            xml[ident] = np.asarray(cargar_imagen(datos / f["imagen"]))
        except (OSError, ImagenInvalidaError) as error:
            meta_xml[ident] = {"localizacion": "error_xml", "detalle_localizacion": str(error)}
        else:
            meta_xml[ident] = {"localizacion": "xml"}
    muestra_manual = preparar_muestra(muestras, xml)
    muestra_manual.to_csv(salida / "annotation_sample.csv", index=False)
    estados = [estado_anotacion(f, anotaciones) for f in muestras]
    filas, espaciales, pares = ([], [], [])
    recortes = {}
    if args.tipo != "automatico":
        filas, espaciales, pares = _experimentar("xml", muestras, xml, meta_xml, metodos, bandas, config, anotaciones)
        preliminar = resumir_cantidad(pd.DataFrame(filas)).query("formato == 'total'")
        mejor = preliminar.sort_values(["tasa_cantidad", "error_absoluto_medio"], ascending=[False, True]).iloc[0]
        elegido = mejor.metodo, mejor.banda
        recortes["xml"] = xml
    else:
        elegido = "S0" if "S0" in metodos else metodos[0], "fila" if "fila" in bandas else bandas[0]

    historico = json.loads(Path("results/stage4b/config.json").read_text(encoding="utf-8"))
    original = historico["detector"]["variantes"]["B3_FILTRO"]
    geometria = historico["geometria_seleccionada"]
    if args.tipo != "xml":
        variante = original["metodo"], ConfigDetector(**original["config"]), original["escalas"]
        automaticas, meta_auto = {}, {}
        for f in muestras:
            entrada, meta = cargar_automatica(datos, f, variante, geometria)
            if entrada is not None:
                automaticas[f["id_imagen"]] = entrada
            meta_auto[f["id_imagen"]] = meta
        nuevas, _, _ = _experimentar("automatico", muestras, automaticas, meta_auto, metodos, bandas, config, {})
        recortes["automatico"] = automaticas
        filas.extend(nuevas)

    detalle = pd.DataFrame(filas)
    resumen = resumir_cantidad(detalle)
    detalle.to_csv(salida / "segmentation_experiments.csv", index=False)
    resumen.to_csv(salida / "segmentation_summary.csv", index=False)
    lista = [fila for tipo, imagenes in recortes.items()
             for fila in exportar_caracteres(salida, imagenes, tipo, muestras, *elegido, config)]
    pd.DataFrame(lista, columns=INVENTARIO).to_csv(salida / "characters.csv", index=False)
    if espaciales:
        pd.DataFrame(espaciales).to_csv(salida / "spatial_experiments.csv", index=False)
        pd.DataFrame(pares).to_csv(salida / "spatial_characters.csv", index=False)
        resumir_espacial(pd.DataFrame(espaciales)).to_csv(salida / "spatial_summary.csv", index=False)
    registro = {
        "ambito": ambito, "ids": [f["id_imagen"] for f in muestras], "tipo": args.tipo,
        "metodos": metodos, "bandas": bandas, "segmentacion": asdict(config), "clases_bpn": CLASES,
        "provisional": {"metodo": elegido[0], "banda": elegido[1],
                        "criterio": "cantidad en XML y error absoluto" if args.tipo != "automatico" else "referencia fija"},
        "detector": {"nombre": "B3_FILTRO", "geometria": geometria, "configuracion": original},
        "anotaciones": {"muestra": len(muestra_manual), "completas": estados.count("completa"),
                        "parciales": estados.count("parcial"), "pendientes": estados.count("pendiente"),
                        "espacial_xml": "calculada" if espaciales else "pendiente", "iou_individual": .5},
        "regiones_exportadas": len(lista),
    }
    especial = historico["detector"]["caso_descartado"]["id_imagen"]
    from scripts.stage5_examples import generar_figuras

    ejemplos = generar_figuras(salida, recortes, xml, detalle, resumen, metodos, elegido, config, especial)
    registro["ejemplos"] = ejemplos
    (salida / "config.json").write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    print(resumen.query("formato == 'total'").round(4).to_string(index=False))
    print(json.dumps({"provisional": registro["provisional"], "anotaciones": registro["anotaciones"],
                      "regiones_exportadas": len(lista)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
