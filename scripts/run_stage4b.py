"""Reproduce la auditoría y la optimización acotada de la etapa 4B."""

import argparse
import os
from pathlib import Path

from src.evaluation import muestras_desarrollo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fase", choices=("todo", "ocr", "detector", "pipeline"), default="todo")
    parser.add_argument("--trabajadores", type=int, default=4, help="Procesos OCR simultáneos")
    parser.add_argument("--salida", type=Path, default=Path("results/stage4b"))
    args = parser.parse_args()
    if not 1 <= args.trabajadores <= 8:
        parser.error("Trabajadores debe estar entre 1 y 8.")

    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    datos, historico = Path("data"), Path("results")
    salida = args.salida
    muestras, ambito = muestras_desarrollo(datos)
    if ambito != "real_desarrollo":
        parser.error("La optimización 4B requiere una partición de desarrollo verificada.")
    for nombre in ("examples", "graphs"):
        (salida / nombre).mkdir(parents=True, exist_ok=True)

    if args.fase in ("todo", "ocr"):
        from scripts.stage4b_ocr import experimentar_ocr

        experimentar_ocr(datos, muestras, historico, salida, args.trabajadores)
    if args.fase in ("todo", "detector"):
        from scripts.stage4b_detection import experimentar_detector

        experimentar_detector(datos, muestras, historico, salida)
    if args.fase in ("todo", "pipeline"):
        from scripts.stage4b_detection import comparar_pipeline

        comparar_pipeline(datos, muestras, historico, salida, args.trabajadores)


if __name__ == "__main__":
    main()
