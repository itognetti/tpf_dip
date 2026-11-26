"""Ablación del detector, diagnóstico por candidato y comparación integrada."""

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.run_detection import _dibujar_caja, mostrar_candidatos
from scripts.stage4b_ocr import medir_ocr, metricas_ocr
from src.dataset import CAJA
from src.detection import (
    ConfigDetector, Deteccion, detectar_patente, extraer_variante, seleccionar_candidato,
)
from src.evaluation import iou, resumir_localizacion
from src.ocr import cargar_imagen
from src.ocr_preparation import PreparacionOCR


def _variantes(base):
    flexible = replace(base, area_relativa=(.00018, .26), rectangularidad_min=.23,
                       ancho_min=30, alto_min=8, aspecto=(1.45, 7), densidad=(.028, .68))
    vertical = replace(flexible, peso_vertical=.18)
    relativo = replace(vertical, kernel_relativo_sobel=(.019, .008))
    return {
        **{f"D{i}": (f"D{i}", base, ()) for i in range(4)},
        "B2_R1200_G3": ("D2", replace(base, max_lado=1200, gauss_kernel=3), ()),
        "B3_R1250_REL": ("D3", replace(base, max_lado=1250, kernel_relativo_sobel=(.019, .008)), ()),
        "B3_S5": ("D3", replace(base, sobel_kernel=5), ()),
        "B3_C13": ("D3", replace(base, kernel_sobel=(13, 3)), ()),
        "B3_ENLACE": ("D3", replace(base, kernel_sobel=(25, 5), iteraciones=2,
                                    kernel_dilatacion=(3, 3), iteraciones_dilatacion=2), ()),
        "B3_FILTRO": ("D3", flexible, ()),
        "B3_CARACTERES": ("D3", replace(flexible, reordenar_k=3, peso_caracteres=.5), ()),
        "B3_VERT": ("D3", vertical, ()),
        "B2_MULTI": ("D2", relativo, (760, 1200)),
        "B3_MULTI": ("D3", relativo, (760, 1200)),
    }


def detectar_variante(foto, variante):
    """Al combinar escalas, expresa cada candidato en píxeles de la foto original."""
    metodo, config, escalas = variante
    if not escalas:
        return detectar_patente(foto, metodo, config)
    efectivos = dict.fromkeys(min(escala, max(foto.shape[:2])) for escala in escalas)
    resultados = [detectar_patente(foto, metodo, replace(config, max_lado=lado)) for lado in efectivos]
    candidatos = sorted((c for paso in resultados for c in paso.candidatos),
                        key=lambda c: (-c.puntaje, c.caja))
    elegido = seleccionar_candidato(candidatos, foto, config)
    motivo = "" if elegido else ("sin_candidato_valido" if candidatos else "sin_contornos")
    return Deteccion(candidatos, elegido, motivo, {}, sum(p.n_contornos for p in resultados))


def diagnosticar_candidatos(referencia, deteccion):
    """Calcula IoU una vez terminada la selección y conserva los rangos efectivos."""
    filas, pos_valida = [], 0
    for rango, candidato in enumerate(deteccion.candidatos, 1):
        valido = not candidato.rechazos
        pos_valida += valido
        filas.append({
            "rango_pre": rango, "rango_post": pos_valida if valido else None,
            "valido": valido, "seleccionado": candidato is deteccion.seleccionado,
            "iou": iou(referencia, candidato.caja), "caja": json.dumps(candidato.caja),
            "puntaje": candidato.puntaje, "escala": candidato.escala,
            "aspecto": candidato.aspecto, "superficie": candidato.superficie,
            "area_relativa": candidato.area_relativa, "rectangularidad": candidato.rectangularidad,
            "densidad": candidato.densidad, "distribucion_vertical": candidato.distribucion_vertical,
            "rechazos": "|".join(candidato.rechazos),
        })
    buenas = [f for f in filas if f["iou"] >= .5]
    validas = [f for f in buenas if f["valido"]]
    seleccionada = next((f for f in filas if f["seleccionado"]), None)
    sel_iou = seleccionada["iou"] if seleccionada else 0.0
    if sel_iou >= .5:
        diagnostico = "correcta"
    elif not buenas:
        diagnostico = "A_no_generada"
    elif not validas:
        diagnostico = "B_descartada"
    else:
        diagnostico = "C_no_seleccionada"
    return filas, {
        "contornos": deteccion.n_contornos, "candidatos_pre": len(filas), "candidatos_post": pos_valida,
        "correcta_pre": bool(buenas), "correcta_post": bool(validas),
        "recall_top3": any(f["rango_post"] <= 3 for f in validas),
        "recall_top5": any(f["rango_post"] <= 5 for f in validas),
        "iou_max_pre": max((f["iou"] for f in filas), default=0.0),
        "iou_max_post": max((f["iou"] for f in filas if f["valido"]), default=0.0),
        "iou": sel_iou, "correcta_050": sel_iou >= .5, "correcta_075": sel_iou >= .75,
        "estado": "detectada" if seleccionada else "sin_deteccion",
        "motivo": deteccion.motivo, "diagnostico": diagnostico,
        "caja_detectada": seleccionada["caja"] if seleccionada else "",
    }


def _caja(fila):
    return tuple(int(fila[k]) for k in CAJA)


def _imagen(datos, fila):
    return np.asarray(cargar_imagen(datos / fila["original"]))


def _graficos_detector(resumen, salida):
    grafico, ejes = plt.subplots(2, 1, figsize=(11, 6.3), layout="constrained", sharex=True)
    tabla = resumen.set_index("metodo")
    tabla[["pre", "post", "top3", "top5", "tasa_iou_050"]].rename(columns={
        "pre": "Antes de filtrar", "post": "Después de filtrar",
        "top3": "Top 3", "top5": "Top 5", "tasa_iou_050": "Selección",
    }).plot.bar(ax=ejes[0], rot=0, width=.85, color=["#b4c9d5", "#7fabb8", "#508a9a", "#3b737e", "#bd6549"])
    ejes[0].set(title="Disponibilidad de cajas correctas y selección", ylabel="Proporción", ylim=(0, 1))
    ejes[0].legend(ncol=3, fontsize=8)
    columnas = ["A_no_generada", "B_descartada", "C_no_seleccionada", "correcta"]
    tabla[columnas].rename(columns=dict(zip(columnas, ("A · ausente", "B · descartada", "C · otra caja", "Correcta")))).plot.bar(
        ax=ejes[1], stacked=True, color=["#c47b64", "#e2b48c", "#889cae", "#3e8173"], rot=45
    )
    ejes[1].set(xlabel="", ylabel="Fotografías", title="Causa de fallo evaluada después de seleccionar")
    ejes[1].legend(ncol=4, fontsize=8)
    grafico.savefig(salida / "graphs/detection_candidates.png", dpi=140)
    plt.close(grafico)


def _ejemplo_descartado(datos, muestras, variantes, seleccion, detalle, salida):
    casos = {f["id_imagen"]: f for f in muestras}
    historico = detalle[detalle.metodo.eq("D3") & detalle.estado.eq("sin_deteccion")].nlargest(1, "iou_max_pre")
    if historico.empty:
        return None
    fila = historico.iloc[0]
    f = casos[fila.id_imagen]
    foto, referencia = _imagen(datos, f), _caja(f)
    anterior = detectar_variante(foto, variantes["D3"])
    mejor = max(anterior.candidatos, key=lambda c: iou(referencia, c.caja), default=None)
    if mejor is None:
        return None
    nueva = detectar_variante(foto, variantes[seleccion])
    fig, (izq, der) = plt.subplots(1, 2, figsize=(9, 4.2), layout="constrained")
    mostrar_candidatos(izq, foto, anterior, referencia, maximos=0)
    _dibujar_caja(izq, mejor.caja, "#ef6c60", "Descartado")
    izq.legend(loc="lower center", bbox_to_anchor=(.5, -.14), ncol=2, frameon=False)
    antiguo_iou = iou(referencia, mejor.caja)
    izq.set_title(f"D3 · candidato IoU {antiguo_iou:.2f}\n{', '.join(mejor.rechazos)}")
    mostrar_candidatos(der, foto, nueva, referencia, maximos=0)
    nuevo_iou = iou(referencia, nueva.seleccionado.caja if nueva.seleccionado else None)
    der.set_title(f"{seleccion} · selección IoU {nuevo_iou:.2f}")
    fig.savefig(salida / "examples/candidato_descartado.png", dpi=140)
    plt.close(fig)
    return {"id_imagen": f["id_imagen"], "iou_pre_anterior": antiguo_iou, "iou_nuevo": nuevo_iou,
            "rechazos_anteriores": list(mejor.rechazos)}


def experimentar_detector(datos, muestras, historico, salida):
    parametros = json.loads((historico / "detection_config.json").read_text(encoding="utf-8"))
    variantes = _variantes(ConfigDetector(**parametros["detector"]))
    filas = []
    with (salida / "candidate_analysis.csv").open("w", encoding="utf-8", newline="") as archivo:
        escritor = None
        for muestra in muestras:
            ident, foto, referencia = muestra["id_imagen"], _imagen(datos, muestra), _caja(muestra)
            for nombre, variante in variantes.items():
                inicio = perf_counter()
                deteccion = detectar_variante(foto, variante)
                ms = (perf_counter() - inicio) * 1000
                candidatos, fila = diagnosticar_candidatos(referencia, deteccion)
                meta = {"id_imagen": ident, "metodo": nombre}
                if candidatos and escritor is None:
                    escritor = csv.DictWriter(archivo, [*meta, *candidatos[0]])
                    escritor.writeheader()
                if candidatos:
                    escritor.writerows(meta | c for c in candidatos)
                filas.append(meta | fila | {"ambito": "real_desarrollo", "ms_deteccion": ms,
                                           "caja_referencia": json.dumps(referencia)})
    detalle = pd.DataFrame(filas)
    referencia_etapa4 = pd.read_csv(historico / "detection_experiments.csv")
    prueba = detalle.merge(referencia_etapa4, on=["id_imagen", "metodo"], suffixes=("", "_hist"))
    if len(prueba) != 4 * len(muestras) or not (
        prueba.caja_detectada.fillna("").eq(prueba.caja_detectada_hist.fillna(""))
        & np.isclose(prueba.iou, prueba.iou_hist)
    ).all():
        raise ValueError("D0–D3 deben conservar la selección histórica en desarrollo.")
    resumen = resumir_localizacion(detalle)
    grupos = detalle.groupby("metodo")
    tasas = grupos[["correcta_pre", "correcta_post", "recall_top3", "recall_top5"]].mean()
    tasas.columns = ["pre", "post", "top3", "top5"]
    causas = pd.crosstab(detalle.metodo, detalle.diagnostico)
    columnas = ["A_no_generada", "B_descartada", "C_no_seleccionada", "correcta"]
    resumen = resumen.join(tasas, on="metodo").join(causas.reindex(columns=columnas, fill_value=0), on="metodo")
    resumen = resumen.set_index("metodo").loc[list(variantes)].reset_index()
    seleccion = resumen.sort_values(["tasa_iou_050", "tasa_iou_075", "iou_medio"], ascending=False,
                                   kind="stable").iloc[0].metodo
    detalle.to_csv(salida / "detection_experiments.csv", index=False)
    resumen.to_csv(salida / "detection_summary.csv", index=False)
    _graficos_detector(resumen, salida)
    especial = _ejemplo_descartado(datos, muestras, variantes, seleccion, detalle, salida)
    registro = {"seleccion": seleccion, "ids": [f["id_imagen"] for f in muestras], "caso_descartado": especial,
                "variantes": {i: {"metodo": m, "config": asdict(c), "escalas": e} for i, (m, c, e) in variantes.items()}}
    (salida / "detector_config.json").write_text(json.dumps(registro, indent=2), encoding="utf-8")
    print(resumen.round(4).to_string(index=False), flush=True)


def _spec(registro):
    return registro["metodo"], ConfigDetector(**registro["config"]), registro["escalas"]


def _extraer(foto, candidato, modo):
    if candidato is None:
        return None, "sin_recorte", 0.0
    inicio = perf_counter()
    imagen, efectiva = extraer_variante(foto, candidato, modo)
    return imagen, efectiva, (perf_counter() - inicio) * 1000


def _cargar_recortes_pipeline(datos, muestras, anterior, nuevo):
    for f in muestras:
        foto, referencia = _imagen(datos, f), _caja(f)
        recortes, tiempos, solapamientos = {}, {}, {}
        for origen, variante, modos in (
            ("anterior", anterior, ("legado",)),
            ("nuevo", nuevo, ("rectangular", "rotado", "perspectiva")),
        ):
            inicio = perf_counter()
            candidato = detectar_variante(foto, variante).seleccionado
            tiempos[origen] = (perf_counter() - inicio) * 1000
            solapamientos[origen] = iou(referencia, candidato.caja if candidato else None)
            for modo in modos:
                recortes[origen, modo] = _extraer(foto, candidato, modo)
        yield f, recortes, tiempos, solapamientos


def resumir_pipeline(detalle):
    filas = []
    for nombre, tabla in detalle.groupby("pipeline", sort=False):
        metricas = metricas_ocr(tabla)
        medida = tabla.iou.notna().any()
        filas.append({
            "pipeline": nombre, "detector": tabla.detector.iloc[0], "ocr": tabla.ocr.iloc[0],
            "psm": tabla.psm.iloc[0], "geometria": tabla.geometria.iloc[0],
            "iou_medio": tabla.iou.mean(),
            "tasa_iou_050": tabla.iou.ge(.50).mean() if medida else None,
            "tasa_iou_075": tabla.iou.ge(.75).mean() if medida else None,
            "ms_total": tabla.ms_total.mean(), **metricas,
        })
    return pd.DataFrame(filas)


def preferencias_pipeline(resumen):
    """Registra las elecciones automáticas al optimizar exactitud o CER."""
    automaticas = resumen[resumen.detector.ne("XML")]
    por_accuracy = automaticas.sort_values(["accuracy", "cer"], ascending=[False, True], kind="stable")
    por_cer = automaticas.sort_values(["cer", "accuracy"], ascending=[True, False], kind="stable")
    return {"por_accuracy": por_accuracy.iloc[0].pipeline, "por_cer": por_cer.iloc[0].pipeline}


def _figura_pipeline(resumen, salida):
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    tabla = resumen.set_index("pipeline")
    for eje, variable, nombre in zip(ejes, ("accuracy", "cer"), ("Coincidencia exacta", "CER micro")):
        tabla[variable].plot.bar(ax=eje, rot=35, color=["#6c94a2"] * 4 + ["#cf9268", "#267868"])
        eje.set(title=nombre, xlabel="")
        eje.grid(axis="y", alpha=.25)
    fig.savefig(salida / "graphs/pipeline_comparison.png", dpi=140)
    plt.close(fig)


def _figura_geometria(datos, recortes, salida):
    modos = ("rectangular", "rotado", "perspectiva")
    elegibles = [
        (f, variantes) for f, variantes, _, solapamientos in recortes
        if solapamientos["nuevo"] >= .5 and variantes["nuevo", "rotado"][1] == "rotado"
    ]
    if not elegibles:
        return
    f, variantes = max(elegibles, key=lambda p: p[1]["nuevo", "perspectiva"][1] == "perspectiva")
    fig, ejes = plt.subplots(2, 2, figsize=(9, 4.8), layout="constrained")
    referencia = cargar_imagen(datos / f["imagen"])
    ejes[0, 0].imshow(referencia)
    ejes[0, 0].set_title("Recorte XML")
    for eje, modo in zip(list(ejes.flat)[1:], modos):
        im, real, _ = variantes["nuevo", modo]
        eje.imshow(im)
        eje.set_title(modo if real == modo else f"{modo} → {real}")
    for eje in ejes.flat:
        eje.axis("off")
    fig.suptitle(f"Mismo candidato · {f['id_imagen']}")
    fig.savefig(salida / "examples/geometria_comparison.png", dpi=140)
    plt.close(fig)


def comparar_pipeline(datos, muestras, historico, salida, trabajadores):
    registro_det = json.loads((salida / "detector_config.json").read_text(encoding="utf-8"))
    registro_ocr = json.loads((salida / "ocr_config.json").read_text(encoding="utf-8"))
    ids = {f["id_imagen"] for f in muestras}
    if ids != set(registro_det["ids"]) or ids != set(registro_ocr["ids"]):
        raise ValueError("Los resultados parciales corresponden a otro conjunto de desarrollo.")

    detector, ganador_ocr = registro_det["seleccion"], registro_ocr["seleccion"]
    variantes = registro_det["variantes"]
    preparaciones = {i: (PreparacionOCR(**registro_ocr["configuraciones"][i]["preparacion"]),
                         registro_ocr["configuraciones"][i]["psm"]) for i in dict.fromkeys(("B0", ganador_ocr))}
    recortes = list(_cargar_recortes_pipeline(datos, muestras, _spec(variantes["D3"]), _spec(variantes[detector])))
    modos = ("rectangular", "rotado", "perspectiva")
    planes = [
        ("anterior", "anterior", "legado", "B0"),
        ("ocr_optimizado", "anterior", "legado", ganador_ocr),
        *[(f"geometria_{modo}", "nuevo", modo, ganador_ocr) for modo in modos],
    ]

    def evaluar(paquete, ensayos):
        f, variantes_recorte, tiempos, solapamientos = paquete
        filas, cache = [], {}
        for nombre, origen, modo, ocr in ensayos:
            roi, efectiva, ms_recorte = variantes_recorte[origen, modo]
            clave = origen, efectiva, ocr
            if clave not in cache:
                cache[clave] = medir_ocr(roi, f["patente"], *preparaciones[ocr])
            lectura = cache[clave]
            filas.append(lectura | {
                "pipeline": nombre, "detector": "D3" if origen == "anterior" else detector,
                "ocr": ocr, "psm": preparaciones[ocr][1], "geometria": modo, "geometria_efectiva": efectiva,
                "id_imagen": f["id_imagen"], "formato": f["formato"], "iou": solapamientos[origen],
                "ms_deteccion": tiempos[origen], "ms_recorte": ms_recorte,
                "ms_total": tiempos[origen] + ms_recorte + lectura["ms_preparacion"] + lectura["ms_ocr"],
            })
        return filas

    with ThreadPoolExecutor(max_workers=trabajadores) as pool:
        filas = [fila for grupo in pool.map(lambda p: evaluar(p, planes), recortes) for fila in grupo]
        mediciones = pd.DataFrame(filas)
        geo_detalle = mediciones[mediciones.pipeline.str.startswith("geometria_")].copy()
        geo_resumen = resumir_pipeline(geo_detalle)
        transformadas = geo_detalle[geo_detalle.geometria_efectiva.isin(("rotado", "perspectiva"))]
        geo_resumen["transformadas"] = (
            transformadas.groupby("pipeline").size().reindex(geo_resumen.pipeline, fill_value=0).to_numpy()
        )
        geometria = geo_resumen.sort_values(["accuracy", "cer"], ascending=[False, True],
                                            kind="stable").iloc[0].geometria
        plan = [("detector_optimizado", "nuevo", geometria, "B0")]
        det_optimizado = [f for grupo in pool.map(lambda p: evaluar(p, plan), recortes) for f in grupo]

    ambos = geo_detalle[geo_detalle.geometria.eq(geometria)].copy()
    ambos["pipeline"] = "ambos_optimizados"
    automatico = pd.concat([mediciones[~mediciones.pipeline.str.startswith("geometria_")],
                             pd.DataFrame(det_optimizado), ambos], ignore_index=True)
    base = pd.read_csv(historico / "detection_ocr_comparison.csv", keep_default_na=False).query("rama == 'automatica'")
    control = automatico[automatico.pipeline.eq("anterior")].merge(base, on="id_imagen", suffixes=("", "_hist"))
    if len(control) != len(muestras) or not control.texto_literal.eq(control.texto_literal_hist).all():
        raise ValueError("El pipeline anterior no reprodujo la salida literal guardada.")
    xml = pd.read_csv(salida / "ocr_experiments.csv", keep_default_na=False)
    referencias = []
    for ident, nombre in (("B0", "xml_anterior"), (ganador_ocr, "xml_optimizado")):
        tabla = xml[xml.config.eq(ident)].copy()
        tabla.confianza = pd.to_numeric(tabla.confianza, errors="coerce")
        tabla = tabla.assign(pipeline=nombre, detector="XML", ocr=ident, geometria="xml", geometria_efectiva="xml",
                             iou=np.nan, ms_deteccion=0., ms_recorte=0., ms_total=tabla.ms_preparacion + tabla.ms_ocr)
        referencias.append(tabla)
    completo = pd.concat([automatico, *referencias], ignore_index=True)
    resumen = resumir_pipeline(completo)
    completo.to_csv(salida / "pipeline_comparison.csv", index=False)
    resumen.to_csv(salida / "pipeline_summary.csv", index=False)
    geo_detalle.to_csv(salida / "geometry_experiments.csv", index=False)
    geo_resumen.to_csv(salida / "geometry_summary.csv", index=False)
    _figura_pipeline(resumen, salida)
    _figura_geometria(datos, recortes, salida)
    registro = {"ambito": "real_desarrollo", "ocr": registro_ocr, "detector": registro_det,
                "geometria_seleccionada": geometria, "trabajadores_pipeline": trabajadores,
                "preferencias_pipeline": preferencias_pipeline(resumen)}
    (salida / "config.json").write_text(json.dumps(registro, indent=2), encoding="utf-8")
    print(resumen.round(4).to_string(index=False), flush=True)
    print(geo_resumen.round(4).to_string(index=False), flush=True)
