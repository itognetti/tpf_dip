"""Recorrido paso a paso del pipeline vigente sobre una fotografía.

Ejecutar con python -m scripts.pipeline_figure [--id ID_IMAGEN | --foto RUTA].
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from src.dataset import CAJA, identificador_imagen
from src.detection import (
    ConfigDetector, Deteccion, detectar_patente, extraer_variante, intensidad_gradiente, sobel,
)
from src.evaluation import iou
from src.ocr import ResultadoOCR, cargar_imagen, reconocer_region
from src.ocr_preparation import PreparacionOCR, preparar_region
from src.segmentation import (
    CaracterNormalizado, ConfigSegmentacion, Segmentacion, normalizar_caracter, segmentar,
)


@dataclass
class Recorrido:
    """Entradas y salidas intermedias de cada paso, en el orden del pipeline."""

    foto: np.ndarray
    deteccion: Deteccion
    gradiente: np.ndarray
    recorte: np.ndarray | None = None
    geometria: str = ""
    entrada_ocr: np.ndarray | None = None
    lectura: ResultadoOCR | None = None
    segmentacion: Segmentacion | None = None
    caracteres: list[CaracterNormalizado] = field(default_factory=list)


def configuracion_vigente(salida_4b: Path, salida_5: Path) -> dict:
    """Lee el detector, la geometría, el OCR y la segmentación elegidos en desarrollo."""
    c4b = json.loads((salida_4b / "config.json").read_text(encoding="utf-8"))
    c5 = json.loads((salida_5 / "config.json").read_text(encoding="utf-8"))
    nombre = c4b["detector"].get("seleccion", "B3_FILTRO")
    detector = c4b["detector"]["variantes"][nombre]
    ocr = c4b["ocr"]["configuraciones"][c4b["ocr"]["seleccion"]]
    return {
        "nombre_detector": nombre,
        "metodo": detector["metodo"], "detector": ConfigDetector(**detector["config"]),
        "geometria": c4b["geometria_seleccionada"], "codigo_ocr": c4b["ocr"]["seleccion"],
        "preparacion": PreparacionOCR(**ocr["preparacion"]), "psm": ocr["psm"],
        "segmentacion": ConfigSegmentacion(**c5["segmentacion"]),
        "metodo_seg": c5["provisional"]["metodo"], "banda": c5["provisional"]["banda"],
        "tesseract": c4b["ocr"].get("tesseract", ""),
    }


def recorrer_pipeline(foto: np.ndarray, vigente: dict, ocr=True) -> Recorrido:
    """Aplica el pipeline usando solo píxeles. La referencia XML queda fuera."""
    config = vigente["detector"]
    deteccion = detectar_patente(foto, vigente["metodo"], config, trazas=True)
    gx = sobel(deteccion.imagenes["filtrada"], config.sobel_kernel)[0]
    recorrido = Recorrido(foto, deteccion, intensidad_gradiente(gx))
    if deteccion.seleccionado is None:
        return recorrido
    recorrido.recorte, recorrido.geometria = extraer_variante(foto, deteccion.seleccionado, vigente["geometria"])
    recorrido.entrada_ocr, _ = preparar_region(recorrido.recorte, vigente["preparacion"])
    if ocr:
        recorrido.lectura = reconocer_region(Image.fromarray(recorrido.entrada_ocr), vigente["psm"])
    recorrido.segmentacion = segmentar(recorrido.recorte, vigente["metodo_seg"], vigente["banda"],
                                       vigente["segmentacion"])
    for x0, y0, x1, y1 in recorrido.segmentacion.cajas:
        try:
            recorrido.caracteres.append(normalizar_caracter(
                recorrido.recorte[y0:y1, x0:x1], vigente["segmentacion"].lado_interno))
        except ValueError:
            continue
    return recorrido


def elegir_ejemplo(salida_4b: Path, salida_5: Path) -> str | None:
    """Elige una foto de desarrollo leída exacta por el pipeline vigente y con cantidad correcta."""
    pipeline = pd.read_csv(salida_4b / "pipeline_comparison.csv", keep_default_na=False, dtype={"id_imagen": str})
    preferido = json.loads((salida_4b / "config.json").read_text(encoding="utf-8"))
    nombre = preferido.get("preferencias_pipeline", {}).get("por_accuracy", "ambos_optimizados")
    filas = pipeline[pipeline["pipeline"].eq(nombre)].copy()
    filas["iou"] = pd.to_numeric(filas["iou"], errors="coerce")
    exactas = filas[filas["exacta"].astype(str).eq("True")]
    segm = pd.read_csv(salida_5 / "segmentation_experiments.csv", dtype={"id_imagen": str})
    cfg5 = json.loads((salida_5 / "config.json").read_text(encoding="utf-8"))["provisional"]
    correctas = segm[segm["tipo_recorte"].eq("automatico") & segm["metodo"].eq(cfg5["metodo"])
                     & segm["banda"].eq(cfg5["banda"]) & segm["cantidad_correcta"].astype(bool)]
    for grupo in (exactas[exactas["id_imagen"].isin(correctas["id_imagen"])], exactas, filas):
        if not grupo.empty:
            return grupo.sort_values("iou", ascending=False).iloc[0]["id_imagen"]
    return None


def _caja(eje, c, color, lw=2.0, escala=1.0, estilo="-"):
    x0, y0, x1, y1 = (v * escala for v in c)
    eje.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=lw, ls=estilo))


def graficar_recorrido(r: Recorrido, vigente: dict, referencia=None, etiqueta="", titulo=""):
    """Dibuja los pasos 0–10. Los paneles intermedios amplían la zona elegida."""
    pasos, elegido, config = r.deteccion.imagenes, r.deteccion.seleccionado, vigente["detector"]
    gris = pasos["gris"]
    escala = gris.shape[1] / r.foto.shape[1]
    validos = [c for c in r.deteccion.candidatos if not c.rechazos]
    if elegido is not None:
        x0, y0, x1, y1 = (round(v * escala) for v in elegido.caja)
        w, h = x1 - x0, y1 - y0
        ventana = (slice(max(0, y0 - 3 * h), min(gris.shape[0], y1 + 3 * h)),
                   slice(max(0, x0 - 2 * w), min(gris.shape[1], x1 + 2 * w)))
    else:
        ventana = (slice(None), slice(None))
    ampliado = " (ampliado)" if elegido is not None else ""

    fig = plt.figure(figsize=(18, 12.5), layout="constrained")
    fig.suptitle(titulo, fontsize=14)
    g = fig.add_gridspec(4, 4, height_ratios=[1.5, 0.9, 0.7, 0.75])

    def panel(pos, img, texto, cmap="gray"):
        eje = fig.add_subplot(pos)
        eje.imshow(img, cmap=cmap)
        eje.set_title(texto, fontsize=10)
        eje.axis("off")
        return eje

    leyenda = "verde = caja XML, rojo = caja detectada" if referencia is not None else "rojo = caja detectada"
    eje = panel(g[0, 0:2], r.foto, f"0 · Fotografía de entrada ({leyenda})")
    if referencia is not None:
        _caja(eje, referencia, "lime", 2.5)
    if elegido is not None:
        _caja(eje, elegido.caja, "red", 1.5)
    panel(g[0, 2], gris[ventana], f"1 · Gris, lado mayor {max(gris.shape)} px{ampliado}")
    panel(g[0, 3], pasos["filtrada"][ventana],
          f"2 · CLAHE límite {config.clahe_limite:g}, grilla {tuple(config.clahe_grilla)}{ampliado}")
    panel(g[1, 0], r.gradiente[ventana], f"3a · |Sobel X| {config.sobel_kernel}×{config.sobel_kernel}{ampliado}",
          "magma")
    panel(g[1, 1], pasos["binaria"][ventana], f"3b · Otsu sobre |Sobel X|{ampliado}")
    panel(g[1, 2], pasos["mascara"][ventana],
          f"4 · Cierre {config.kernel_sobel[0]}×{config.kernel_sobel[1]}{ampliado}")
    puntaje = f"puntaje {elegido.puntaje:.2f}" if elegido is not None else "sin selección"
    if elegido is not None and elegido.evidencia is not None:
        puntaje += f" · caracteres {elegido.evidencia:.2f}"
    if elegido is not None and elegido.origen:
        puntaje += f" · respaldo: {elegido.origen}"
    nombre = vigente.get("nombre_detector", "")
    eje = panel(g[1, 3], gris, f"5–7 · {nombre} · {len(r.deteccion.candidatos)} contornos → "
                f"{len(validos)} admitidos\namarillo = admitido · rojo = elegido\n{puntaje}")
    for c in validos:
        if c is not elegido:
            _caja(eje, c.caja, "yellow", 1, escala)
    if elegido is not None:
        _caja(eje, elegido.caja, "red", 2, escala)

    if r.recorte is None:
        fig.add_subplot(g[2:, :]).axis("off")
        fig.text(.5, .25, f"Sin recorte automático: {r.deteccion.motivo}", ha="center", fontsize=13)
        return fig

    solape = f" · IoU {iou(referencia, elegido.caja):.2f}" if referencia is not None else ""
    panel(g[2, 0], r.recorte, f"8 · Recorte {r.geometria} {r.recorte.shape[1]}×{r.recorte.shape[0]}{solape}")
    prep = vigente["preparacion"]
    if r.lectura is None:
        leido = "OCR omitido"
    else:
        leido = f"Tesseract: '{r.lectura.texto_original.strip()}' · formato válido: {r.lectura.formato_valido}"
    panel(g[2, 1], r.entrada_ocr, f"9A · Entrada OCR ({vigente['codigo_ocr']}: {prep.filtrado}, "
          f"PSM {vigente['psm']})\n{leido}")
    seg = r.segmentacion
    esperado = f" (esperadas {len(etiqueta)})" if etiqueta else ""
    eje = panel(g[2, 2], seg.binaria, f"9B · {vigente['metodo_seg']} banda {vigente['banda']}: "
                f"{seg.binaria.shape[0]} px, CLAHE, Otsu\n{len(seg.cajas_trabajo)} regiones{esperado}")
    for c in seg.cajas_trabajo:
        _caja(eje, c, "cyan", 1.5)
    eje = panel(g[2, 3], r.recorte, "9B · Banda (naranja) y cajas de caracteres")
    _caja(eje, seg.banda, "orange", 1.5, estilo="--")
    for c in seg.cajas:
        _caja(eje, c, "cyan", 1.5)

    sub = g[3, :].subgridspec(1, max(1, len(r.caracteres)))
    for i, caracter in enumerate(r.caracteres):
        eje = fig.add_subplot(sub[0, i])
        eje.imshow(caracter.lienzo, cmap="gray", vmin=0, vmax=255)
        real = f" (real '{etiqueta[i]}')" if i < len(etiqueta) else ""
        eje.set_title(f"10 · región {i + 1}{real}\n28×28 → 784 float32", fontsize=9)
        eje.axis("off")
    if not r.caracteres:
        fig.add_subplot(sub[0, 0]).axis("off")
    return fig


def figura_desarrollo(datos: Path, salida_4b: Path, salida_5: Path, id_imagen=None, ocr=True):
    """Arma la figura para una foto de desarrollo y devuelve también un resumen."""
    filas = pd.read_csv(datos / "ground_truth.csv", keep_default_na=False, dtype={"id_imagen": str})
    id_imagen = id_imagen or elegir_ejemplo(salida_4b, salida_5)
    fila = filas.set_index("id_imagen").loc[id_imagen]
    if fila["particion"] != "desarrollo":
        raise ValueError("La figura solo admite fotografías de desarrollo.")
    vigente = configuracion_vigente(salida_4b, salida_5)
    referencia = tuple(int(fila[k]) for k in CAJA)
    r = recorrer_pipeline(np.asarray(cargar_imagen(datos / fila["original"])), vigente, ocr)
    version = f" · Tesseract {vigente['tesseract']}" if vigente["tesseract"] else ""
    fig = graficar_recorrido(r, vigente, referencia, fila["patente"],
                             f"Pipeline paso a paso · foto {id_imagen} · patente {fila['patente']}{version}")
    elegido = r.deteccion.seleccionado
    resumen = {
        "id_imagen": id_imagen, "patente": fila["patente"],
        "iou": round(iou(referencia, elegido.caja), 4) if elegido else 0.0,
        "contornos": len(r.deteccion.candidatos),
        "admitidos": sum(not c.rechazos for c in r.deteccion.candidatos),
        "texto_ocr": r.lectura.texto_normalizado if r.lectura else None,
        "regiones": len(r.segmentacion.cajas) if r.segmentacion else 0,
    }
    return fig, resumen


def figura_foto(ruta: Path, datos: Path, salida_4b: Path, salida_5: Path, ocr=True):
    """Arma la figura para una foto cualquiera, sin etiqueta ni caja de referencia.

    Rechaza las fotos de evaluación final reconocidas por su huella de píxeles.
    """
    imagen = cargar_imagen(ruta)
    ident = identificador_imagen(imagen)
    filas = pd.read_csv(datos / "ground_truth.csv", keep_default_na=False, dtype={"id_imagen": str})
    particion = filas.set_index("id_imagen")["particion"].get(ident, "")
    if particion == "evaluacion_final":
        raise ValueError("La foto pertenece a la evaluación final reservada.")
    vigente = configuracion_vigente(salida_4b, salida_5)
    r = recorrer_pipeline(np.asarray(imagen), vigente, ocr)
    version = f" · Tesseract {vigente['tesseract']}" if vigente["tesseract"] else ""
    fig = graficar_recorrido(r, vigente, titulo=f"Pipeline paso a paso · {Path(ruta).name}{version}")
    elegido = r.deteccion.seleccionado
    resumen = {
        "foto": Path(ruta).as_posix(), "id_imagen": ident, "particion": particion or "fuera_del_experimento",
        "caja": list(elegido.caja) if elegido else None, "motivo": r.deteccion.motivo,
        "contornos": len(r.deteccion.candidatos),
        "admitidos": sum(not c.rechazos for c in r.deteccion.candidatos),
        "texto_literal": r.lectura.texto_original.strip() if r.lectura else None,
        "texto_ocr": r.lectura.texto_normalizado if r.lectura else None,
        "formato_valido": r.lectura.formato_valido if r.lectura else None,
        "regiones": len(r.segmentacion.cajas) if r.segmentacion else 0,
    }
    return fig, resumen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    origen = parser.add_mutually_exclusive_group()
    origen.add_argument("--id", help="ID de una foto de desarrollo. Por defecto se elige un caso exacto.")
    origen.add_argument("--foto", type=Path, help="Ruta a una foto nueva, sin etiqueta.")
    parser.add_argument("--salida", type=Path, help="PNG de salida. Por defecto en results/graphs/.")
    args = parser.parse_args()
    rutas = Path("data"), Path("results/stage4b"), Path("results/stage5")
    if args.foto:
        try:
            fig, resumen = figura_foto(args.foto, *rutas)
        except (OSError, ValueError) as error:
            parser.exit(1, f"{error}\n")
        args.salida = args.salida or Path("results/graphs") / f"pipeline_{args.foto.stem}.png"
    else:
        fig, resumen = figura_desarrollo(*rutas, args.id)
        args.salida = args.salida or Path("results/graphs/pipeline_paso_a_paso.png")
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.salida, dpi=110)
    print(json.dumps(resumen | {"salida": args.salida.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
