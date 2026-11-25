"""Métricas de preprocesamiento, localización y reconocimiento."""

import json
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image

from .dataset import formato_patente, leer_etiquetas
from .detection import Caja, Candidato, extraer_patente
from .ocr import MODOS_PSM, ImagenInvalidaError, cargar_imagen, reconocer_patente
from .preprocessing import METODOS, aplicar_metodo, parametros_metodo

REFERENCIAS_SINTETICAS = {
    "examples/demo_antigua_ABC123.png": "ABC123",
    "examples/demo_mercosur_AB123CD.png": "AB123CD",
}


def distancia_levenshtein(esperado: str, obtenido: str) -> int:
    """Cuenta inserciones, borrados y sustituciones de coste unitario."""
    anterior = list(range(len(obtenido) + 1))
    for i, origen in enumerate(esperado, start=1):
        actual = [i]
        for j, destino in enumerate(obtenido, start=1):
            actual.append(min(actual[-1] + 1, anterior[j] + 1, anterior[j - 1] + (origen != destino)))
        anterior = actual
    return anterior[-1]


def cer(esperado: str, obtenido: str) -> float:
    """Normaliza la distancia por la longitud de una referencia no vacía."""
    if not esperado:
        raise ValueError("El CER requiere una referencia no vacía.")
    return distancia_levenshtein(esperado, obtenido) / len(esperado)


def muestras_desarrollo(datos: Path) -> tuple[list[dict], str]:
    """Selecciona etiquetas verificadas de desarrollo o la demostración sintética."""
    filas = leer_etiquetas(datos / "ground_truth.csv")
    reservadas = {
        fila["patente"] for fila in filas
        if fila["estado"] == "verificada" and fila["particion"] in {"evaluacion_final", "excluida"}
    }
    desarrollo = [
        fila for fila in filas if fila["estado"] == "verificada" and fila["particion"] == "desarrollo"
    ]
    if any(fila["patente"] in reservadas for fila in desarrollo):
        raise ValueError("Una patente verificada cruza particiones. Corregí el CSV antes de evaluar.")
    if desarrollo:
        return desarrollo, "real_desarrollo"
    sinteticas = [
        {"id_imagen": Path(recorte).stem, "imagen": recorte, "patente": texto,
         "formato": formato_patente(texto), "particion": "demostracion"}
        for recorte, texto in REFERENCIAS_SINTETICAS.items()
    ]
    return sinteticas, "sintetico_demostracion"


def evaluar_muestras(datos: Path, muestras: list[dict], ambito: str, psm=7) -> pd.DataFrame:
    """Evalúa M0–M9 con PSM fijo y registra todos los intentos."""
    if not isinstance(psm, int) or psm not in MODOS_PSM:
        raise ValueError("Seleccioná PSM 7 u 8.")

    filas = []
    for muestra in muestras:
        esperado = muestra["patente"]
        formato = formato_patente(esperado)
        try:
            entrada = np.asarray(cargar_imagen(datos / muestra["imagen"]))
            error_carga = None
        except (OSError, ImagenInvalidaError) as error:
            error_carga = str(error)

        for metodo in METODOS:
            fila = {
                "ambito": ambito, "psm": psm, "idioma": "eng",
                "id_imagen": muestra["id_imagen"], "imagen": muestra["imagen"], "formato": formato,
                "metodo": metodo, "parametros": json.dumps(parametros_metodo(metodo), ensure_ascii=False),
                "polaridad_invertida": None, "esperado": esperado, "texto_literal": None,
                "texto_normalizado": None, "confianza": None, "exacta": False, "formato_valido": False,
                "estado": "error_carga" if error_carga else "ok", "error": error_carga,
                "ms_preprocesamiento": None, "ms_ocr": None,
            }
            if error_carga is None:
                _reconocer_metodo(entrada, fila)
            obtenido = fila["texto_normalizado"] or ""
            distancia = distancia_levenshtein(esperado, obtenido)
            fila.update(levenshtein=distancia, cer=distancia / len(esperado))
            if fila["ms_preprocesamiento"] is not None and fila["ms_ocr"] is not None:
                fila["ms_total"] = fila["ms_preprocesamiento"] + fila["ms_ocr"]
            else:
                fila["ms_total"] = None
            filas.append(fila)
    return pd.DataFrame(filas)


def _reconocer_metodo(entrada, fila):
    inicio = perf_counter()
    try:
        procesada, invertida = aplicar_metodo(entrada, fila["metodo"])
        imagen = Image.fromarray(procesada)
        fila["polaridad_invertida"] = invertida
    except (ValueError, cv2.error) as error:
        fila.update(estado="error_preprocesamiento", error=str(error))
        return
    finally:
        fila["ms_preprocesamiento"] = 1000 * (perf_counter() - inicio)

    inicio = perf_counter()
    try:
        lectura = reconocer_patente(imagen, fila["psm"])
    except (pytesseract.TesseractError, OSError, RuntimeError) as error:
        fila.update(estado="error_ocr", error=str(error))
        return
    finally:
        fila["ms_ocr"] = 1000 * (perf_counter() - inicio)
    prediccion = lectura.texto_normalizado
    fila.update(
        texto_literal=lectura.texto_original, texto_normalizado=prediccion,
        confianza=lectura.confianza, formato_valido=lectura.formato_valido,
        exacta=prediccion == fila["esperado"], estado="ok" if prediccion else "sin_texto",
    )


def resumir_experimentos(detalle: pd.DataFrame) -> pd.DataFrame:
    """Agrega por ámbito, PSM y método, con el total y cada formato."""
    if detalle.empty:
        return pd.DataFrame()
    filas = []
    for (ambito, psm, metodo), grupo in detalle.groupby(["ambito", "psm", "metodo"]):
        for formato, tabla in [("total", grupo), *grupo.groupby("formato")]:
            filas.append({
                "ambito": ambito, "psm": psm, "metodo": metodo, "formato": formato,
                "imagenes": len(tabla), "aciertos": int(tabla["exacta"].sum()),
                "accuracy": tabla["exacta"].mean(),
                "cer": tabla["levenshtein"].sum() / tabla["esperado"].str.len().sum(),
                "tasa_formato_valido": tabla["formato_valido"].mean(),
                "confianza_media": tabla["confianza"].mean(),
                "con_confianza": int(tabla["confianza"].count()),
                "sin_texto": int(tabla["estado"].eq("sin_texto").sum()),
                "errores": int(tabla["estado"].str.startswith("error_").sum()),
                "ms_preprocesamiento": tabla["ms_preprocesamiento"].mean(),
                "ms_ocr": tabla["ms_ocr"].mean(), "ms_total": tabla["ms_total"].mean(),
            })
    return pd.DataFrame(filas)


def graficar_resumen(resumen: pd.DataFrame):
    """Muestra el resultado global y los resultados por formato para un PSM."""
    total = resumen[resumen["formato"].eq("total")].set_index("metodo")
    formatos = resumen[resumen["formato"].ne("total")]
    fig, ejes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for eje, columna, titulo in zip(
        ejes, ("accuracy", "cer", "ms_total"), ("Accuracy de patente", "CER micro", "Procesamiento medio"),
    ):
        valores = total[columna]
        eje.bar(total.index, valores, color="#28736b")
        eje.set_title(titulo)
        eje.set_ylabel("ms" if columna == "ms_total" else "Proporción")
        if columna == "accuracy":
            eje.set_ylim(0, 1.08)
        elif columna == "cer":
            _ajustar_eje_cer(eje, valores.max())
        eje.grid(axis="y", alpha=0.25)

    fig_formatos, ejes_formatos = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for eje, columna, titulo in zip(ejes_formatos, ("accuracy", "cer"), ("Accuracy por formato", "CER por formato")):
        tabla = formatos.pivot(index="metodo", columns="formato", values=columna)
        tabla.plot.bar(ax=eje, rot=0, color=["#28736b", "#ac613f"][:len(tabla.columns)])
        eje.set_title(titulo)
        eje.set_xlabel("")
        eje.set_ylabel("Proporción")
        eje.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
        if columna == "accuracy":
            eje.set_ylim(0, 1.08)
        else:
            _ajustar_eje_cer(eje, tabla.max().max())
        eje.grid(axis="y", alpha=0.25)
    return fig, fig_formatos


def _ajustar_eje_cer(eje, maximo):
    eje.set_ylim(0, max(0.05, maximo * 1.1))
    if maximo == 0:
        eje.text(0.5, 0.5, "CER = 0 en todos los métodos", ha="center", transform=eje.transAxes)


def graficar_metodos(entrada: np.ndarray):
    """Compara la entrada con los resultados espaciales M0–M9."""
    fig, ejes = plt.subplots(5, 2, figsize=(10, 9), layout="constrained")
    for eje, metodo in zip(ejes.flat, METODOS):
        salida, invertida = aplicar_metodo(entrada, metodo)
        eje.imshow(salida, cmap="gray", vmin=0, vmax=255)
        eje.set_title(f"{metodo} · polaridad invertida" if invertida else metodo)
        eje.axis("off")
    return fig


def exportar_experimentos(detalle, resumen, salida: Path):
    """Guarda CSV y gráficos compactos de una ejecución experimental."""
    graficos = salida / "graphs"
    graficos.mkdir(parents=True, exist_ok=True)
    detalle.to_csv(salida / "preprocessing_experiments.csv", index=False, encoding="utf-8")
    resumen.to_csv(salida / "preprocessing_summary.csv", index=False, encoding="utf-8")
    figuras = graficar_resumen(resumen)
    for figura, nombre in zip(figuras, ("preprocessing_comparison", "preprocessing_by_format")):
        figura.savefig(graficos / f"{nombre}.png", dpi=130)
    return figuras


def iou(referencia: Caja, detectada: Caja | None) -> float:
    """Intersección sobre unión de dos rectángulos semiabiertos."""
    x0, y0, x1, y1 = referencia
    if x0 >= x1 or y0 >= y1:
        raise ValueError("IoU requiere cajas de área positiva.")
    if detectada is None:
        return 0.0
    u0, v0, u1, v1 = detectada
    if u0 >= u1 or v0 >= v1:
        raise ValueError("IoU requiere cajas de área positiva.")
    inter = max(0, min(x1, u1) - max(x0, u0)) * max(0, min(y1, v1) - max(y0, v0))
    union = (x1 - x0) * (y1 - y0) + (u1 - u0) * (v1 - v0) - inter
    return inter / union


def resumir_localizacion(detalle: pd.DataFrame) -> pd.DataFrame:
    """Mantiene los intentos sin detección en los denominadores globales."""
    filas = []
    for (ambito, metodo), grupo in detalle.groupby(["ambito", "metodo"]):
        existe = grupo["estado"].eq("detectada")
        filas.append({
            "ambito": ambito, "metodo": metodo, "imagenes": len(grupo),
            "tasa_iou_050": grupo["correcta_050"].mean(),
            "tasa_iou_075": grupo["correcta_075"].mean(),
            "iou_medio": grupo["iou"].mean(), "iou_con_deteccion": grupo.loc[existe, "iou"].mean(),
            "detecciones_fallidas": int((~existe).sum()), "ms_medio": grupo["ms_deteccion"].mean(),
        })
    return pd.DataFrame(filas)


def comparar_ocr_localizacion(
    referencia: np.ndarray | None, foto: np.ndarray | None, candidato: Candidato | None,
    *, esperado: str, metodo_espacial="M0", psm=7,
) -> list[dict]:
    """Aplica el mismo OCR a ambas ramas; un fallo automático computa texto vacío."""
    formato_patente(esperado)
    if metodo_espacial not in METODOS or psm not in MODOS_PSM:
        raise ValueError("Configuración de OCR experimental inválida.")
    filas = []
    for rama in ("referencia", "automatica"):
        fila = {
            "rama": rama, "metodo": metodo_espacial, "psm": psm, "esperado": esperado,
            "tipo_recorte": "", "motivo_geometria": "", "estado": "ok", "error": "",
            "texto_literal": None, "texto_normalizado": None,
            "confianza": None, "formato_valido": False, "exacta": False,
            "polaridad_invertida": None, "ms_preprocesamiento": None, "ms_ocr": None,
        }
        recorte = referencia
        if rama == "referencia":
            if referencia is None:
                fila["estado"] = "error_carga_referencia"
            else:
                fila["tipo_recorte"] = "xml"
        elif foto is None or candidato is None:
            recorte = None
            fila["estado"] = "sin_deteccion"
        else:
            try:
                extraccion = extraer_patente(foto, candidato)
            except (ValueError, cv2.error) as error:
                recorte = None
                fila.update(estado="error_extraccion", error=str(error))
            else:
                fila["motivo_geometria"] = extraccion.motivo
                if extraccion.rectificada is not None:
                    recorte = extraccion.rectificada
                    fila["tipo_recorte"] = "rectificada"
                else:
                    recorte = extraccion.rectangular
                    fila["tipo_recorte"] = "rectangular"
        if recorte is not None:
            _reconocer_metodo(recorte, fila)
        distancia = distancia_levenshtein(esperado, fila["texto_normalizado"] or "")
        fila.update(levenshtein=distancia, cer=distancia / len(esperado))
        filas.append(fila)
    return filas


def resumir_ocr_localizacion(detalle: pd.DataFrame) -> pd.DataFrame:
    """Compara accuracy y CER micro sobre los mismos IDs verificados."""
    filas = []
    for rama, grupo in detalle.groupby("rama", sort=False):
        filas.append({
            "rama": rama, "imagenes": len(grupo), "aciertos": int(grupo["exacta"].sum()),
            "accuracy": grupo["exacta"].mean(),
            "cer": grupo["levenshtein"].sum() / grupo["esperado"].str.len().sum(),
            "sin_deteccion": int(grupo["estado"].eq("sin_deteccion").sum()),
            "sin_texto": int(grupo["estado"].eq("sin_texto").sum()),
            "errores": int(grupo["estado"].str.startswith("error_").sum()),
        })
    return pd.DataFrame(filas)


def graficar_localizacion(resumen: pd.DataFrame):
    """Compara tasas por umbral e IoU con y sin condicionar a una detección."""
    fig, ejes = plt.subplots(1, 2, figsize=(9.5, 3.8), layout="constrained")
    datos = resumen.set_index("metodo")
    for eje, columnas, titulo in [
        (ejes[0], {"tasa_iou_050": "IoU ≥ 0,50", "tasa_iou_075": "IoU ≥ 0,75"}, "Tasa de localización"),
        (ejes[1], {"iou_medio": "Todas", "iou_con_deteccion": "Con detección"}, "IoU promedio"),
    ]:
        datos[list(columnas)].rename(columns=columnas).plot.bar(
            ax=eje, rot=0, color=["#28736b", "#ac613f"], width=0.72
        )
        eje.set(title=titulo, xlabel="", ylim=(0, 1))
        eje.grid(axis="y", alpha=0.25)
        eje.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.suptitle(f"{resumen.iloc[0]['ambito']} · {resumen.iloc[0]['imagenes']} imágenes por método")
    return fig
