"""Ejecuta el pipeline clásico sobre una foto, una carpeta o la muestra de defensa."""

import argparse
import csv
import json
from pathlib import Path

from src.pipeline import RAIZ, cargar_configuracion, preparar_motor, procesar_foto

EXTENSIONES = {".jpg", ".jpeg", ".jpe", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"}
CARPETAS = ("detected", "cropped", "processed", "characters", "normalized", "visualizations")
MUESTRA = RAIZ / "data/examples/preentrega.json"


def fotos_entrada(entrada):
    if entrada.is_dir():
        return sorted((p for p in entrada.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONES),
                      key=lambda ruta: (ruta.name.casefold(), ruta.name))
    return [entrada]


def exportar_resumen(salida, filas, registro):
    csv_destino = salida / "resultados.csv"
    with csv_destino.open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)
    lineas = [
        f"Pipeline clásico · perfil {registro['perfil_ocr']} · {len(filas)} imágenes",
        "Los estados de formato y segmentación describen las salidas automáticas.",
    ]
    for fila in filas:
        confianza = fila["confianza"]
        lineas.extend((
            "", fila["archivo"],
            f"  Detección: {fila['estado_deteccion']} · caja {fila['caja_detectada'] or '—'} · "
            f"geometría {fila['geometria_efectiva'] or '—'}",
            f"  OCR: {fila['estado_ocr']} · literal {fila['texto_literal']!r} · "
            f"normalizado {fila['texto_normalizado']!r} · confianza "
            f"{format(confianza, '.2f') if confianza is not None else '—'}",
            f"  Segmentación: {fila['estado_segmentacion']} · regiones {fila['regiones_segmentadas']}",
            f"  Normalización: {fila['estado_normalizacion']} · caracteres {fila['caracteres_normalizados']}",
            f"  Tiempo de procesamiento: {fila['ms_procesamiento']:.1f} ms",
        ))
        if fila["motivo_fallo"]:
            lineas.append(f"  Incidencia: {fila['motivo_fallo']}")
    (salida / "resultados.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    (salida / "config.json").write_text(json.dumps(registro, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--entrada", type=Path, help="Fotografía o carpeta de fotografías")
    grupo.add_argument("--demo", action="store_true", help="Muestra ilustrativa fija de desarrollo")
    parser.add_argument("--perfil", choices=("exactitud", "cer"), default="exactitud")
    parser.add_argument("--salida", type=Path, default=Path("results/demo"))
    args = parser.parse_args(argv)

    rutas = ([RAIZ / caso["foto"] for caso in json.loads(MUESTRA.read_text(encoding="utf-8"))]
             if args.demo else fotos_entrada(args.entrada))
    if not rutas:
        parser.error("La carpeta no contiene fotografías de las extensiones admitidas.")
    destino = args.salida.resolve()
    carpetas_exportadas = {(destino / carpeta).resolve() for carpeta in CARPETAS}
    metadatos = {(destino / nombre).resolve() for nombre in ("resultados.csv", "resultados.txt", "config.json")}
    if ((args.entrada is not None and destino == args.entrada.resolve())
            or any(ruta.parent.resolve() in carpetas_exportadas or ruta.resolve() in metadatos for ruta in rutas)):
        parser.error("Elegí una salida separada de las fotografías de entrada.")

    # Matplotlib usa un backend sin ventanas incluso cuando el script lo ejecuta un lote.
    import matplotlib

    matplotlib.use("Agg")
    from src.pipeline_visualization import exportar_imagenes

    config = cargar_configuracion(args.perfil)
    version, problema = preparar_motor()
    registro = config.registro()
    registro.update(tesseract=version, problema_motor=problema, modo="demo" if args.demo else "usuario",
                    archivos=[ruta.name for ruta in rutas])
    if problema:
        print(problema)
    for carpeta in CARPETAS:
        (args.salida / carpeta).mkdir(parents=True, exist_ok=True)

    filas = []
    for indice, ruta in enumerate(rutas, 1):
        resultado = procesar_foto(ruta, config, problema)
        nombre = f"{indice:03d}_{ruta.stem[:90]}"
        fila = resultado.fila
        fila.update(id_salida=nombre)
        try:
            archivos = exportar_imagenes(resultado, args.salida, nombre)
        except OSError as error:
            fila["motivo_fallo"] = " | ".join(filter(None, (fila["motivo_fallo"], f"Exportación: {error}")))
            archivos = {}
        fila.update({f"ruta_{carpeta}": archivos.get(carpeta) for carpeta in
                     ("detected", "cropped", "processed", "visualizations")})
        filas.append(fila)
        print(f"{ruta.name} | {fila['estado_deteccion']} | OCR {fila['estado_ocr']} | "
              f"regiones {fila['regiones_segmentadas']}")

    exportar_resumen(args.salida, filas, registro)
    print(f"Resultados en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
