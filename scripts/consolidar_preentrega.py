"""Reúne resultados históricos de desarrollo y evidencia de la demostración."""

import argparse
import json
import shutil
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.run_segmentation import resumir_espacial
from src.character_annotations import (
    caja_anotada, estado_anotacion, evaluar_espacial, leer_anotaciones, referencias_xml,
)
from src.dataset import leer_etiquetas
from src.evaluation import iou
from src.ocr import cargar_imagen
from src.pipeline import CONFIGURACION, RAIZ
from src.preprocessing import METODOS, parametros_metodo

RESULTADOS = RAIZ / "results"
ANOTACIONES = RAIZ / "data/character_annotations.csv"


def estado_individual(anotaciones=None):
    """Cuenta el avance actual y marca superposiciones de cajas para revisión."""
    datos = RAIZ / "data"
    if anotaciones is None:
        anotaciones = leer_anotaciones(ANOTACIONES)
    muestra = pd.read_csv(RESULTADOS / "stage5/annotation_sample.csv", dtype={"id_imagen": str})
    seleccion = set(muestra.id_imagen)
    desarrollo = {f["id_imagen"]: f for f in leer_etiquetas(datos / "ground_truth.csv")
                  if f["particion"] == "desarrollo" and f["estado"] == "verificada"}
    if not seleccion <= set(desarrollo):
        raise ValueError("La muestra individual contiene fotos fuera de desarrollo.")
    avance = {"patentes_muestra": len(seleccion), "cajas_humanas": 0, "completas": 0, "parciales": 0,
              "pendientes": 0, "ids_completas": [], "cajas_para_revisar": []}
    for ident in sorted(seleccion):
        f = desarrollo[ident]
        estado = estado_anotacion(f, anotaciones)
        avance[{"completa": "completas", "parcial": "parciales", "pendiente": "pendientes"}[estado]] += 1
        if estado == "completa":
            avance["ids_completas"].append(ident)
        cajas = [(indice, caja) for (foto, indice), caja in anotaciones.items()
                 if foto == ident and caja["estado"] == "anotado"]
        avance["cajas_humanas"] += len(cajas)
        if len(cajas) > 1:
            ancho, alto = cargar_imagen(datos / f["imagen"]).size
            for (i, a), (j, b) in combinations(cajas, 2):
                solapamiento = iou(caja_anotada(a, ancho, alto), caja_anotada(b, ancho, alto))
                if solapamiento >= .50:
                    avance["cajas_para_revisar"].append({"id_imagen": ident, "indices": [i, j],
                                                         "solapamiento": round(solapamiento, 4)})
    return avance


def evaluar_anotaciones(identificadores, anotaciones=None):
    """Evalúa las cajas históricas XML contra anotaciones humanas completas de la muestra."""
    datos = RAIZ / "data"
    if anotaciones is None:
        anotaciones = leer_anotaciones(ANOTACIONES)
    desarrollo = {f["id_imagen"]: f for f in leer_etiquetas(datos / "ground_truth.csv")
                  if f["particion"] == "desarrollo" and f["estado"] == "verificada"}
    historico = pd.read_csv(RESULTADOS / "stage5/segmentation_experiments.csv",
                            dtype={"id_imagen": str}, keep_default_na=False)
    filas, pares = [], []
    for ident in identificadores:
        muestra = desarrollo[ident]
        ancho, alto = cargar_imagen(datos / muestra["imagen"]).size
        cajas = referencias_xml(muestra, anotaciones, ancho, alto)
        intentos = historico[historico.id_imagen.eq(ident) & historico.tipo_recorte.eq("xml")]
        for fila in intentos.itertuples():
            resultado, asociaciones = evaluar_espacial(
                cajas, json.loads(fila.cajas), json.loads(fila.banda_caja) if fila.banda_caja else None,
            )
            clave = {"id_imagen": ident, "metodo": fila.metodo, "banda": fila.banda}
            filas.append(clave | resultado)
            pares.extend(clave | par for par in asociaciones)
    return pd.DataFrame(filas), pd.DataFrame(pares)


def leer_historico():
    """Obtiene solo filas históricas reales y comprueba coincidencia de la partición."""
    datos = RAIZ / "data"
    ids = {f["id_imagen"] for f in leer_etiquetas(datos / "ground_truth.csv")
           if f["particion"] == "desarrollo" and f["estado"] == "verificada"}
    grupos = (
        ("stage3_desarrollo/preprocessing_experiments.csv", "stage3_desarrollo/preprocessing_summary.csv"),
        ("stage4b/detection_experiments.csv", "stage4b/detection_summary.csv"),
        ("stage4b/pipeline_comparison.csv", "stage4b/pipeline_summary.csv"),
        ("stage5/segmentation_experiments.csv", "stage5/segmentation_summary.csv"),
    )
    tablas = []
    for detalle, resumen in grupos:
        observado = pd.read_csv(RESULTADOS / detalle, usecols=["id_imagen"], dtype=str)
        if set(observado.id_imagen) != ids:
            raise ValueError(f"Los ID guardados en {detalle} difieren de la división de desarrollo.")
        tabla = pd.read_csv(RESULTADOS / resumen)
        if "formato" in tabla.columns:
            tabla = tabla[tabla.formato.eq("total")].drop(columns="formato")
        if not tabla.imagenes.eq(len(ids)).all():
            raise ValueError(f"El denominador en {resumen} no coincide con la división.")
        tablas.append(tabla)
    return tablas, ids


def graficar(tablas, salida):
    espacio, detector, ocr, segmentos = tablas
    fig, ejes = plt.subplots(2, 2, figsize=(12.8, 8.5), layout="constrained")
    fig.suptitle("Resultados históricos · 58 fotografías de desarrollo", fontsize=15)
    ejes[0, 0].bar(espacio.metodo, espacio.aciertos, color="#287987")
    ejes[0, 0].set(title="M0–M9 · OCR de recortes XML", ylabel="Patentes exactas")
    basicos = detector[detector.metodo.isin(("D0", "D1", "D2", "D3", "B3_FILTRO"))]
    ejes[0, 1].bar(basicos.metodo, basicos.tasa_iou_050 * 100, color="#287987")
    ejes[0, 1].set(title="Detectores de referencia y configuración elegida", ylabel="IoU ≥ 0,50 (%)", ylim=(0, 100))
    etiquetas = ["D3 + B0", "D3 + PSM6", "B3 + B0", "B3 + PSM6", "XML + B0", "XML + PSM6"]
    orden = ["anterior", "ocr_optimizado", "detector_optimizado", "ambos_optimizados",
             "xml_anterior", "xml_optimizado"]
    comparacion = ocr.set_index("pipeline").loc[orden]
    ejes[1, 0].barh(etiquetas[::-1], comparacion.cer.iloc[::-1], color="#895d2c")
    ejes[1, 0].set(title="Etapa 4B · CER micro", xlabel="Ediciones por carácter esperado")
    cuentas = segmentos[segmentos.banda.eq("fila")].pivot(index="metodo", columns="tipo_recorte",
                                                          values="cantidad_correcta")
    cuentas[["xml", "automatico"]].rename(columns={"xml": "XML", "automatico": "Automático"}).plot.bar(
        ax=ejes[1, 1], rot=0, color=["#287987", "#895d2c"])
    ejes[1, 1].set(title="Banda por gradiente · conteo de regiones", ylabel="Patentes con cantidad esperada",
                   xlabel="", ylim=(0, 58))
    fig.savefig(salida, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, default=RESULTADOS / "preentrega")
    parser.add_argument("--demo", type=Path, default=RESULTADOS / "demo")
    args = parser.parse_args(argv)
    plt.switch_backend("Agg")
    salida = args.salida
    (salida / "graphs").mkdir(parents=True, exist_ok=True)
    (salida / "examples").mkdir(exist_ok=True)
    tablas, ids = leer_historico()
    espacio, detector, ocr, segmentos = tablas
    pd.concat([
        espacio.assign(ensayo="preprocesamiento_xml", variante=espacio.metodo, tiempo_medio_ms=espacio.ms_total),
        detector.assign(ensayo="localizacion", variante=detector.metodo, tiempo_medio_ms=detector.ms_medio),
        ocr.assign(ensayo="ocr_etapa4b", variante=ocr.pipeline, tiempo_medio_ms=ocr.ms_total),
        segmentos.assign(ensayo="segmentacion", variante=segmentos.metodo + "/" + segmentos.banda,
                         tiempo_medio_ms=segmentos.ms_promedio),
    ])[["ensayo", "variante", "tipo_recorte", "detector", "ocr", "psm", "geometria", "imagenes",
        "aciertos", "accuracy", "cer", "sin_texto", "tasa_iou_050", "tasa_iou_075", "iou_medio",
        "cantidad_correcta", "tasa_cantidad", "tiempo_medio_ms"]].to_csv(salida / "metricas.csv", index=False)

    configuracion = json.loads(CONFIGURACION.read_text(encoding="utf-8"))
    filas = [
        {"etapa": "espacial", "metodo": m, "descripcion": json.dumps(parametros_metodo(m)) if m != "M0" else "RGB original"}
        for m in METODOS
    ]
    filas += [{"etapa": "localizacion", "metodo": m, "descripcion": descripcion}
              for m, descripcion in {
                  "D0": "Canny", "D1": "Canny y cierre", "D2": "Gaussiano, Sobel X, Otsu y cierre",
                  "D3": "CLAHE, Sobel X, Otsu y cierre",
                  "B3_FILTRO": "D3 con filtros geométricos ampliados; conserva el puntaje",
              }.items()]
    filas += [{"etapa": "segmentacion", "metodo": m, "descripcion": d}
              for m, d in (("S0", "Otsu y contornos"), ("S1", "Adaptativo y componentes conexas"),
                           ("S2", "Otsu y proyección vertical"))]
    pd.DataFrame(filas).to_csv(salida / "metodos.csv", index=False)
    perfiles = configuracion["perfiles"]
    pd.DataFrame([
        {"perfil": nombre, "detector": "B3_FILTRO", "geometria": configuracion["geometria"],
         "psm": ajuste["psm"], **ajuste["preparacion"], "segmentacion": "S0/fila"}
        for nombre, ajuste in perfiles.items()
    ]).to_csv(salida / "configuraciones.csv", index=False)
    (salida / "configuraciones.json").write_text(json.dumps(configuracion, indent=2) + "\n", encoding="utf-8")
    anotaciones = leer_anotaciones(ANOTACIONES)
    avance = estado_individual(anotaciones)
    if avance["completas"]:
        detalle, pares = evaluar_anotaciones(avance["ids_completas"], anotaciones)
        if not detalle.empty:
            detalle.to_csv(salida / "espacial_xml_detalle.csv", index=False)
            pares.to_csv(salida / "espacial_xml_pares.csv", index=False)
            resumir_espacial(detalle).to_csv(salida / "espacial_xml_resumen.csv", index=False)
    (salida / "anotaciones.json").write_text(json.dumps(avance, indent=2) + "\n", encoding="utf-8")
    nota_espacial = ("El CSV espacial se limita a las cajas XML completas de la muestra."
                     if avance["completas"] else "La evaluación espacial requiere recortes con anotación completa.")
    (salida / "limitaciones.md").write_text(
        "# Limitaciones y resultados pendientes\n\n"
        "- Las cifras corresponden a las 58 fotos usadas para el desarrollo. La evaluación final sigue reservada.\n"
        "- La localización se evalúa con la caja de patente. Una superposición alta admite caracteres cortados.\n"
        "- Se observaron OCR vacíos, sustituciones, fusiones, fragmentos y regiones de marco.\n"
        "- La coincidencia de cantidad solo mide el conteo.\n"
        f"- Anotación individual: {avance['completas']} completas, {avance['parciales']} parciales y "
        f"{avance['pendientes']} pendientes en la muestra de {avance['patentes_muestra']}. "
        f"{nota_espacial}\n"
        "- La demostración ilustra casos seleccionados. Sus salidas no se agregan como evaluación.\n",
        encoding="utf-8",
    )
    graficar(tablas, salida / "graphs/comparacion.png")

    muestra = pd.read_json(RAIZ / "data/examples/preentrega.json", dtype={"id_imagen": str})
    if not set(muestra.id_imagen) <= ids:
        raise ValueError("La demostración contiene un caso ajeno a desarrollo.")
    registro_demo = args.demo / "config.json"
    if registro_demo.is_file():
        pasada = json.loads(registro_demo.read_text(encoding="utf-8"))
        if (pasada["modo"] == "demo" and pasada["perfil_ocr"] == "exactitud"
                and pasada["archivos"] == [Path(f).name for f in muestra.foto]):
            corrida = pd.read_csv(args.demo / "resultados.csv", keep_default_na=False)
            rutas = []
            for ruta in corrida.ruta_visualizations:
                destino = salida / "examples" / Path(ruta).name
                if ruta:
                    shutil.copyfile(args.demo / ruta, destino)
                rutas.append(destino.relative_to(salida).as_posix() if ruta else "")
            muestra["visualizacion"] = rutas
    muestra.to_csv(salida / "ejemplos.csv", index=False)
    print(f"Resultados consolidados en {salida}")


if __name__ == "__main__":
    main()
