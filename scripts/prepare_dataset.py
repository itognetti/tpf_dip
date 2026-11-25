"""Importa un export Pascal VOC. Ejecutar con python -m scripts.prepare_dataset."""

import argparse
import json
from pathlib import Path

from src.dataset import FUENTE_MASTER, preparar_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=Path("data/raw/roboflow_argentina"))
    parser.add_argument("--limite", type=int, default=125)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--fuente", default=FUENTE_MASTER)
    parser.add_argument("--coordenadas", choices=("roboflow", "voc"), default="roboflow")
    args = parser.parse_args()
    try:
        resumen = preparar_dataset(
            args.entrada, Path("data"), Path("results/dataset_import.csv"),
            limite=args.limite, semilla=args.semilla, fuente=args.fuente,
            convencion=args.coordenadas,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
