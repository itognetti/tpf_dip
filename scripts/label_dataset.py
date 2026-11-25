"""Muestra recortes y guarda sus transcripciones humanas en cada paso."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.dataset import CAJA, COLUMNAS, dividir_experimento, etiquetar_fila, guardar_csv, leer_etiquetas
from src.ocr import ImagenInvalidaError, cargar_imagen


def _mostrar(datos, fila):
    imagen = cargar_imagen(datos / fila["imagen"])
    incluir_original = bool(fila["original"])
    fig, ejes = plt.subplots(1, 2 if incluir_original else 1, figsize=(10, 4), squeeze=False)
    eje = ejes[0, -1]
    eje.imshow(imagen)
    eje.set_title(f"Recorte {fila['id_imagen'] or fila['imagen']}")
    eje.axis("off")
    if incluir_original:
        vista = ejes[0, 0]
        vista.axis("off")
        try:
            foto = cargar_imagen(datos / fila["original"])
            x, y, derecha, abajo = (int(fila[k]) for k in CAJA)
        except (OSError, ImagenInvalidaError, ValueError) as error:
            vista.set_title("Fotografía no disponible")
            print(error)
        else:
            vista.imshow(foto, extent=(0, foto.width, foto.height, 0))
            vista.add_patch(Rectangle((x, y), derecha - x, abajo - y, fill=False, edgecolor="orange"))
            vista.set_title("Fotografía de origen")
    fig.tight_layout()
    plt.show(block=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    acciones = parser.add_mutually_exclusive_group()
    acciones.add_argument("--corregir", metavar="ID", help="Revisar una etiqueta por ID o ruta de recorte")
    acciones.add_argument("--dividir", action="store_true", help="Fijar la división agrupada 70/30")
    parser.add_argument("--semilla", type=int, default=42, help="Semilla para la división")
    args = parser.parse_args()
    ruta = Path("data/ground_truth.csv")
    filas = leer_etiquetas(ruta)

    if args.dividir:
        try:
            desarrollo, final = dividir_experimento(filas, args.semilla)
            guardar_csv(ruta, filas, COLUMNAS)
        except (OSError, ValueError) as error:
            parser.exit(1, f"{error}\n")
        print(f"División fijada. Desarrollo: {desarrollo}. Evaluación final: {final}.")
        return

    if args.corregir:
        pendientes = [f for f in filas if args.corregir in (f["id_imagen"], f["imagen"])]
        if len(pendientes) != 1:
            parser.error("El ID o la ruta debe corresponder a un único recorte.")
    else:
        pendientes = [f for f in filas if f["estado"] == "pendiente"]
    if not pendientes:
        print("No hay recortes pendientes de etiquetado.")
        return

    print("Cerrá la ventana de cada recorte para escribir. Opciones: :ilegible, :fuera, :saltar, :salir.")
    try:
        for fila in pendientes:
            print(f"{fila['id_imagen']} · {fila['imagen']} · {fila['estado']} · {fila['patente']}")
            try:
                _mostrar(ruta.parent, fila)
            except (OSError, ImagenInvalidaError) as error:
                print(f"No se pudo mostrar el recorte: {error}")
                continue
            while True:
                texto = input("Patente o comando: ").strip()
                if texto.casefold() == ":salir":
                    return
                if texto.casefold() == ":saltar":
                    break
                observaciones = input("Observación (opcional): ").strip()
                try:
                    etiquetar_fila(fila, texto, observaciones, filas)
                except ValueError as error:
                    print(error)
                    continue
                guardar_csv(ruta, filas, COLUMNAS)
                print(f"Guardado: {fila['estado']}.")
                if fila["particion"] == "excluida":
                    print("La corrección cruza particiones. El grupo queda excluido de ambas.")
                break
    except (EOFError, KeyboardInterrupt):
        print("\nSesión finalizada. Se conservaron los cambios guardados.")


if __name__ == "__main__":
    main()
