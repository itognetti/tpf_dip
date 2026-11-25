"""Reproduce la comparación espacial con un único PSM por ejecución."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytesseract

from src.evaluation import (
    evaluar_muestras, exportar_experimentos, graficar_metodos, muestras_desarrollo, resumir_experimentos,
)
from src.ocr import ImagenInvalidaError, cargar_imagen, configurar_tesseract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psm", type=int, choices=(7, 8), default=7)
    parser.add_argument("--salida", type=Path, default=Path("results"))
    args = parser.parse_args()
    datos = Path("data")
    configurar_tesseract()
    try:
        version = pytesseract.get_tesseract_version()
        if "eng" not in pytesseract.get_languages():
            parser.error("Tesseract requiere los datos de idioma eng.")
        muestras, ambito = muestras_desarrollo(datos)
    except (OSError, pytesseract.TesseractError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    detalle = evaluar_muestras(datos, muestras, ambito, args.psm)
    resumen = resumir_experimentos(detalle)
    exportar_experimentos(detalle, resumen, args.salida)
    for muestra in muestras:
        try:
            entrada = np.asarray(cargar_imagen(datos / muestra["imagen"]))
        except (OSError, ImagenInvalidaError):
            continue
        figura = graficar_metodos(entrada)
        figura.savefig(args.salida / "graphs/preprocessing_examples.png", dpi=130)
        break
    plt.close("all")
    print(f"{ambito} · Tesseract {version} · PSM {args.psm} · {len(detalle)} intentos.")
    print(resumen[resumen["formato"].eq("total")].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
