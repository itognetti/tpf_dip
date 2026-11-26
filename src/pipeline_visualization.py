"""Figuras y archivos para seguir la ejecución automática sin referencias."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


def _mostrar(eje, titulo, imagen=None, mensaje="Etapa no completada"):
    eje.set_title(titulo, fontsize=10)
    eje.axis("off")
    if imagen is None:
        eje.text(.5, .5, mensaje, transform=eje.transAxes, ha="center", va="center", wrap=True, fontsize=9)
    else:
        eje.imshow(imagen, cmap="gray", vmin=0, vmax=255,
                   extent=(0, imagen.shape[1], imagen.shape[0], 0))
        eje.set_xlim(0, imagen.shape[1])
        eje.set_ylim(imagen.shape[0], 0)


def _caja(eje, caja, color, ancho=2):
    x0, y0, x1, y1 = caja
    eje.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=ancho))


def _regiones(eje, resultado):
    regiones = resultado.segmentacion
    _mostrar(eje, "12 · Regiones candidatas", resultado.recorte)
    if regiones is None or resultado.recorte is None:
        return
    for indice, caja in enumerate(regiones.cajas, 1):
        _caja(eje, caja, "#11b98d", 1.5)
        eje.text(caja[0], caja[1], str(indice), color="white", va="bottom",
                 fontsize=7, bbox={"facecolor": "#166b57", "pad": 1})
    if not regiones.cajas:
        eje.set_title("12 · Sin regiones candidatas", fontsize=10)


def figura_pipeline(resultado):
    """Muestra trazas reales y estados de una fotografía procesada."""
    fig, matriz = plt.subplots(4, 3, figsize=(11.8, 10), layout="constrained")
    ejes = matriz.ravel()
    fila = resultado.fila
    fig.suptitle(f"{fila['archivo']} · recorrido automático · perfil {fila['perfil_ocr']}", fontsize=15)
    trazas = resultado.deteccion.imagenes if resultado.deteccion else {}
    _mostrar(ejes[0], "1 · Fotografía original", resultado.foto, "Imagen no disponible")
    _mostrar(ejes[1], "2 · Escala de grises", trazas.get("gris"))
    _mostrar(ejes[2], "3 · Contraste local CLAHE", trazas.get("filtrada"))
    _mostrar(ejes[3], "4 · Sobel X · magnitud normalizada", trazas.get("gradiente"))
    _mostrar(ejes[4], "5 · Otsu y cierre morfológico", trazas.get("mascara"))
    _mostrar(ejes[5], "6 · Candidatos geométricos", resultado.foto)
    _mostrar(ejes[6], "7 · Localización seleccionada", resultado.foto)
    if resultado.deteccion is not None:
        for candidato in resultado.deteccion.candidatos[:8]:
            _caja(ejes[5], candidato.caja, "#e45f53" if candidato.rechazos else "#3296d2", 1.25)
        if fila["caja_detectada"] is not None:
            _caja(ejes[5], fila["caja_detectada"], "#1be7ac")
            _caja(ejes[6], fila["caja_detectada"], "#1be7ac")
        else:
            ejes[6].set_title("7 · Sin candidato seleccionado", fontsize=10)
    _mostrar(ejes[7], f"8 · Recorte ({fila['geometria_efectiva'] or 'pendiente'})", resultado.recorte)
    texto = fila["texto_normalizado"] or "∅"
    estado_ocr = fila["estado_ocr"].replace("_", " ")
    _mostrar(ejes[8], f"9–10 · OCR · PSM {fila['psm']}\n{texto[:38]} · {estado_ocr}", resultado.entrada_ocr)
    _mostrar(ejes[9], "11 · Binarización de la banda",
             resultado.segmentacion.binaria if resultado.segmentacion else None)
    _regiones(ejes[10], resultado)
    cuadros = [n.lienzo for _, _, n in resultado.caracteres if n is not None]
    mosaico = None
    if cuadros:
        columnas = min(8, len(cuadros))
        mosaico = np.zeros(((len(cuadros) + columnas - 1) // columnas * 32, columnas * 32), np.uint8)
        for indice, cuadro in enumerate(cuadros):
            y, x = divmod(indice, columnas)
            mosaico[y * 32:y * 32 + 28, x * 32:x * 32 + 28] = cuadro
    _mostrar(ejes[11], "13 · Normalización a 28 × 28", mosaico, "Sin caracteres normalizados")
    return fig


def exportar_imagenes(resultado, salida, nombre):
    """Guarda archivos de etapas completadas y devuelve rutas relativas a la salida."""
    rutas = {}
    for carpeta, imagen in (("cropped", resultado.recorte), ("processed", resultado.entrada_ocr)):
        if imagen is not None:
            ruta = salida / carpeta / f"{nombre}.png"
            Image.fromarray(imagen).save(ruta)
            rutas[carpeta] = ruta.relative_to(salida).as_posix()
    if resultado.fila["caja_detectada"] is not None:
        fig, eje = plt.subplots(figsize=(7, 4), layout="constrained")
        _mostrar(eje, "Caja seleccionada · B3_FILTRO", resultado.foto)
        _caja(eje, resultado.fila["caja_detectada"], "#1be7ac")
        ruta = salida / "detected" / f"{nombre}.png"
        fig.savefig(ruta, dpi=120)
        plt.close(fig)
        rutas["detected"] = ruta.relative_to(salida).as_posix()
    for indice, mascara, normalizado in resultado.caracteres:
        ruta = salida / "characters" / f"{nombre}_{indice:02d}.png"
        Image.fromarray(mascara).save(ruta)
        if normalizado is not None:
            lugar = salida / "normalized" / ruta.name
            Image.fromarray(normalizado.lienzo).save(lugar)
    ruta = salida / "visualizations" / f"{nombre}.png"
    fig = figura_pipeline(resultado)
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    rutas["visualizations"] = ruta.relative_to(salida).as_posix()
    return rutas
