"""Anotación manual de caracteres en una ventana Matplotlib sobre recortes XML."""

import argparse
import csv
from math import ceil, floor
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector

from src.character_annotations import caja_anotada, estado_anotacion, guardar_anotaciones, leer_anotaciones
from src.dataset import CAJA
from src.evaluation import muestras_desarrollo
from src.ocr import cargar_imagen


def caja_seleccion(p1, p2, ancho, alto):
    """Convierte los extremos dibujados a coordenadas enteras y semiabiertas."""
    x0, x1 = sorted((p1[0], p2[0]))
    y0, y1 = sorted((p1[1], p2[1]))
    caja = max(0, floor(x0)), max(0, floor(y0)), min(ancho, ceil(x1)), min(alto, ceil(y1))
    return caja if caja[0] < caja[2] and caja[1] < caja[3] else None


def registrar(anotaciones, muestra, indice, caja):
    """Asocia la caja dibujada con el carácter humano elegido en la ventana."""
    if not 1 <= indice <= len(muestra["patente"]):
        raise ValueError("La posición individual está fuera de la transcripción.")
    ident = muestra["id_imagen"]
    fila = {"id_imagen": ident, "indice": indice, "caracter": muestra["patente"][indice - 1],
            "estado": "anotado" if caja else "omitido", "observacion": "" if caja else "Sin límite claro"}
    fila.update(zip(CAJA, caja or ("",) * 4))
    anotaciones[ident, indice] = fila


def _revisar(recorte, muestra, anotaciones, salida):
    ident, palabra = muestra["id_imagen"], muestra["patente"]
    fig, eje = plt.subplots(figsize=(11, 4.5), layout="constrained")
    eje.imshow(recorte, extent=(0, recorte.width, recorte.height, 0), interpolation="nearest")
    eje.axis("off")
    libre = next((i for i in range(1, len(palabra) + 1) if (ident, i) not in anotaciones), 1)
    indice, salir = libre, True
    marcas = []

    def actualizar():
        for marca in marcas:
            marca.remove()
        marcas.clear()
        estados = []
        for i, letra in enumerate(palabra, 1):
            fila = anotaciones.get((ident, i))
            estado = fila["estado"] if fila else "pendiente"
            estados.append(f"{letra}/{estado[0]}")
            if estado == "anotado":
                x0, y0, x1, y1 = caja_anotada(fila, recorte.width, recorte.height)
                color = "orange" if i == indice else "lime"
                marca = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color)
                eje.add_patch(marca)
                marcas.extend([marca, eje.text(x0, y0, f"{i} {letra}", color=color, fontsize=9)])
        eje.set_title(f"{ident} · {palabra} · posición {indice}: {palabra[indice - 1]}\n"
                      + "   ".join(estados))
        fig.canvas.draw_idle()

    def guardar(caja):
        nonlocal indice
        registrar(anotaciones, muestra, indice, caja)
        guardar_anotaciones(salida, anotaciones)
        indice = min(len(palabra), indice + 1)
        actualizar()

    def seleccionar(inicio, fin):
        if any(v is None for v in (inicio.xdata, inicio.ydata, fin.xdata, fin.ydata)):
            return
        caja = caja_seleccion((inicio.xdata, inicio.ydata), (fin.xdata, fin.ydata), recorte.width, recorte.height)
        if caja:
            guardar(caja)

    def teclado(evento):
        nonlocal indice, salir
        if evento.key == "left":
            indice = max(1, indice - 1)
        elif evento.key == "right":
            indice = min(len(palabra), indice + 1)
        elif evento.key == "x":
            guardar(None)
        elif evento.key == "backspace":
            anotaciones.pop((ident, indice), None)
            guardar_anotaciones(salida, anotaciones)
        elif evento.key in ("enter", "escape"):
            salir = evento.key == "escape"
            plt.close(fig)
            return
        actualizar()

    selector = RectangleSelector(eje, seleccionar, button=[1], minspanx=1, minspany=1)
    fig.canvas.mpl_connect("key_press_event", teclado)
    actualizar()
    plt.show(block=True)
    selector.set_active(False)
    return salir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    acciones = parser.add_mutually_exclusive_group()
    acciones.add_argument("--lista", action="store_true", help="Lista la muestra y el progreso en consola")
    acciones.add_argument("--id", help="Abre un ID de desarrollo para anotar o corregir")
    args = parser.parse_args()
    datos = Path("data")
    salida = datos / "character_annotations.csv"
    muestras, ambito = muestras_desarrollo(datos)
    if ambito != "real_desarrollo":
        parser.error("Se necesita una partición verificada de desarrollo.")
    dev = {f["id_imagen"]: f for f in muestras}
    anotaciones = leer_anotaciones(salida)
    if args.id:
        if args.id not in dev:
            parser.error("El ID debe pertenecer a desarrollo.")
        escogidas = [dev[args.id]]
    else:
        ruta = Path("results/stage5/annotation_sample.csv")
        if not ruta.is_file():
            parser.error("Ejecutá scripts.run_segmentation para preparar la muestra.")
        with ruta.open(encoding="utf-8", newline="") as archivo:
            escogidas = [dev[f["id_imagen"]] for f in csv.DictReader(archivo) if f["id_imagen"] in dev]
    if args.lista:
        for f in escogidas:
            print(f["id_imagen"], f["formato"], estado_anotacion(f, anotaciones))
        return
    print("Arrastrá con el botón izquierdo. Flechas: posición. X: omitir. "
          "Retroceso: borrar. Enter: siguiente foto. Esc o cerrar: salir. Se guarda cada cambio.")
    for f in escogidas:
        if args.id or any((f["id_imagen"], i) not in anotaciones for i in range(1, len(f["patente"]) + 1)):
            if _revisar(cargar_imagen(datos / f["imagen"]), f, anotaciones, salida):
                break


if __name__ == "__main__":
    main()
