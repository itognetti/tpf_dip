"""Preparación y etiquetado del conjunto real de recortes de referencia."""

import csv
import hashlib
import random
import re
import shutil
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from xml.etree import ElementTree

from .ocr import ImagenInvalidaError, cargar_imagen, es_formato_valido

FUENTE_MASTER = "https://universe.roboflow.com/xenon-bo5e9/patentes-argentinas-master/dataset/1"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CAJA = ("xmin", "ymin", "xmax", "ymax")
COLUMNAS = (
    "imagen", "patente", "id_imagen", "original", "formato", "procedencia",
    "ruta_fuente", "particion_origen", "particion", "estado", "observaciones",
    "ancho_original", "alto_original", *CAJA, "coordenadas_fuente",
)
COLUMNAS_AUDITORIA = (
    "ruta_fuente", "anotacion", "procedencia", "particion_origen", "estado", "detalle",
    "id_imagen", "ancho_original", "alto_original", *CAJA, "coordenadas_fuente", "gris_medio",
)


def leer_etiquetas(ruta: Path) -> list[dict]:
    """Acepta el CSV anterior de dos columnas y completa los campos nuevos."""
    if not ruta.exists():
        return []
    with ruta.open(encoding="utf-8-sig", newline="") as archivo:
        lector = csv.DictReader(archivo)
        if not {"imagen", "patente"}.issubset(lector.fieldnames or ()):
            raise ValueError("El CSV debe incluir las columnas imagen y patente.")
        return [dict.fromkeys(COLUMNAS, "") | fila for fila in lector]


@contextmanager
def _archivo_atomico(ruta: Path, **opciones):
    temporal = NamedTemporaryFile(dir=ruta.parent, delete=False, **opciones)
    reemplazo = Path(temporal.name)
    try:
        with temporal:
            yield temporal
        reemplazo.replace(ruta)
    finally:
        reemplazo.unlink(missing_ok=True)


def guardar_csv(ruta: Path, filas: list[dict], columnas: tuple) -> None:
    """Actualiza el CSV de forma atómica para guardar el progreso por imagen."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    campos = list(dict.fromkeys([*columnas, *(campo for fila in filas for campo in fila)]))
    with _archivo_atomico(ruta, mode="w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)


def _copiar_original(origen: Path, destino: Path) -> None:
    with origen.open("rb") as fuente, _archivo_atomico(destino) as archivo:
        shutil.copyfileobj(fuente, archivo)


def leer_caja_voc(xml: Path, archivo: str, dimensiones: tuple, convencion="roboflow") -> tuple:
    """Devuelve una única caja válida en coordenadas de recorte semiabiertas."""
    anotacion = ElementTree.parse(xml).getroot()
    if anotacion.tag != "annotation" or anotacion.findtext("filename") != archivo:
        raise ValueError("La anotación no corresponde al nombre de la imagen.")
    ancho, alto = dimensiones
    medidas = tuple(int(anotacion.findtext(f"size/{lado}", "")) for lado in ("width", "height"))
    if medidas != dimensiones:
        raise ValueError(f"Dimensiones XML {medidas} distintas de la imagen {dimensiones}.")
    objetos = [obj for obj in anotacion.findall("object")
               if obj.findtext("name", "").strip().casefold() == "patente"]
    if len(objetos) != 1:
        raise ValueError(f"Se esperaba una sola caja de clase patente. Encontradas: {len(objetos)}.")
    caja = tuple(int(objetos[0].findtext(f"bndbox/{clave}", "")) for clave in CAJA)
    if convencion == "voc":
        izquierda, arriba, derecha, abajo = caja
        caja = izquierda - 1, arriba - 1, derecha, abajo
    elif convencion != "roboflow":
        raise ValueError(f"Convención de coordenadas desconocida: {convencion}.")
    izquierda, arriba, derecha, abajo = caja
    if not (0 <= izquierda < derecha <= ancho and 0 <= arriba < abajo <= alto):
        raise ValueError(f"Caja inválida {caja} para {ancho} × {alto}.")
    return caja


def _inspeccionar(raiz: Path, fuente: str, convencion: str) -> list[dict]:
    archivos = sorted(p for p in raiz.rglob("*") if p.is_file())
    imagenes = [p for p in archivos if p.suffix.lower() in EXTENSIONES]
    xmls = {p for p in archivos if p.suffix.lower() == ".xml"}
    xml_por_base = defaultdict(list)
    for ruta in xmls:
        xml_por_base[ruta.parent, ruta.stem.casefold()].append(ruta)
    imagenes_por_base = Counter((p.parent, p.stem.casefold()) for p in imagenes)

    def ficha(ruta):
        relativa = ruta.relative_to(raiz)
        return {
            "ruta_fuente": relativa.as_posix(), "procedencia": fuente,
            "particion_origen": relativa.parts[0] if len(relativa.parts) > 1 else "",
            "coordenadas_fuente": convencion,
        }

    filas, vistos, xml_pareados = [], {}, set()
    for ruta in imagenes:
        fila = ficha(ruta)
        filas.append(fila)
        base = ruta.parent, ruta.stem.casefold()
        candidatos = xml_por_base[base]
        xml_pareados.update(candidatos)
        if len(candidatos) != 1 or imagenes_por_base[base] != 1:
            fila.update(estado="sin_xml" if not candidatos else "pareo_ambiguo")
            continue
        xml = candidatos[0]
        fila["anotacion"] = xml.relative_to(raiz).as_posix()
        try:
            imagen = cargar_imagen(ruta)
        except (OSError, ImagenInvalidaError) as error:
            fila.update(estado="imagen_invalida", detalle=str(error))
            continue
        fila.update(ancho_original=imagen.width, alto_original=imagen.height)
        try:
            caja = leer_caja_voc(xml, ruta.name, imagen.size, convencion)
        except (OSError, ValueError, ElementTree.ParseError) as error:
            fila.update(estado="anotacion_invalida", detalle=str(error))
            continue
        fila.update(zip(CAJA, caja))
        huella = hashlib.sha256(f"{imagen.width}x{imagen.height}:".encode())
        huella.update(imagen.tobytes())
        identificador = huella.hexdigest()[:20]
        fila["id_imagen"] = identificador
        if identificador in vistos:
            fila.update(estado="duplicada", detalle=vistos[identificador])
            continue
        vistos[identificador] = fila["ruta_fuente"]
        gris = imagen.crop(caja).convert("L")
        gris_medio = sum(i * n for i, n in enumerate(gris.histogram())) / (gris.width * gris.height)
        fila.update(estado="elegible", gris_medio=round(gris_medio, 2))

    for ruta in archivos:
        if ruta in xmls and ruta not in xml_pareados:
            filas.append(ficha(ruta) | {"estado": "xml_sin_imagen"})
        elif ruta not in xmls and ruta.suffix.lower() not in EXTENSIONES | {".txt", ".md"}:
            filas.append(ficha(ruta) | {"estado": "archivo_invalido"})
    return filas


def _seleccionar(filas, limite, semilla):
    grupos = defaultdict(list)
    for fila in filas:
        if fila["estado"] == "elegible":
            ancho = fila["xmax"] - fila["xmin"]
            gris = fila["gris_medio"]
            estrato = (ancho >= 120, ancho >= 240, gris >= 85, gris >= 170)
            grupos[estrato].append(fila)
    azar = random.Random(semilla)
    claves = sorted(grupos)
    azar.shuffle(claves)
    pendientes = []
    for clave in claves:
        azar.shuffle(grupos[clave])
        pendientes.append(deque(grupos[clave]))
    elegidas = []
    while pendientes and len(elegidas) < limite:
        for grupo in pendientes:
            if len(elegidas) == limite:
                break
            elegidas.append(grupo.popleft())
        pendientes = [grupo for grupo in pendientes if grupo]
    return elegidas


def preparar_dataset(
    raiz: Path, datos: Path, auditoria: Path, *, limite=125, semilla=42,
    fuente=FUENTE_MASTER, convencion="roboflow",
) -> dict:
    """Inspecciona, selecciona y copia un subconjunto reproducible del export."""
    if not raiz.is_dir():
        raise FileNotFoundError(f"No existe el directorio del export: {raiz}")
    if limite < 1:
        raise ValueError("El límite debe ser mayor que cero.")
    entrada = raiz.resolve()
    directorios = [(datos / nombre).resolve() for nombre in ("original", "cropped")]
    archivos_salida = [(datos / "ground_truth.csv").resolve(), auditoria.resolve()]
    if any(p.is_relative_to(entrada) for p in [*directorios, *archivos_salida]) or any(
        entrada.is_relative_to(p) for p in directorios
    ):
        raise ValueError("Los destinos del proyecto deben quedar fuera del export de entrada.")
    inventario = _inspeccionar(raiz, fuente, convencion)
    if not inventario:
        raise ValueError("El export está vacío.")
    ruta_csv = datos / "ground_truth.csv"
    etiquetas = leer_etiquetas(ruta_csv)
    previas = {fila["id_imagen"]: fila for fila in etiquetas if fila["id_imagen"]}
    elegidas = _seleccionar(inventario, limite, semilla)
    nuevos = {fila["id_imagen"] for fila in elegidas} - previas.keys()
    if nuevos and any(fila["particion"] for fila in previas.values()):
        raise ValueError("La división experimental ya está fijada. Usá un conjunto independiente.")

    for fila in inventario:
        if fila["estado"] == "elegible":
            fila["estado"] = "no_seleccionada"
            if fila["id_imagen"] in previas:
                fila["estado"] = "importada_previa"

    origenes = datos / "original"
    recortes = datos / "cropped"
    origenes.mkdir(parents=True, exist_ok=True)
    recortes.mkdir(parents=True, exist_ok=True)
    for fila in elegidas:
        identificador = fila["id_imagen"]
        fuente_imagen = raiz / fila["ruta_fuente"]
        anterior = previas.get(identificador)
        original = anterior["original"] if anterior else f"original/{identificador}{fuente_imagen.suffix.lower()}"
        recorte = anterior["imagen"] if anterior else f"cropped/{identificador}.png"
        if not (datos / original).is_file():
            _copiar_original(fuente_imagen, datos / original)
        if not (datos / recorte).is_file():
            imagen = cargar_imagen(fuente_imagen)
            roi = imagen.crop(tuple(fila[k] for k in CAJA))
            with _archivo_atomico(datos / recorte) as archivo:
                roi.save(archivo, format="PNG")
        if not anterior:
            ficha = {k: fila[k] for k in COLUMNAS if k in fila and k != "estado"}
            etiquetas.append(ficha | {
                "original": original, "imagen": recorte, "estado": "pendiente",
            })
            fila["estado"] = "importada"

    guardar_csv(ruta_csv, etiquetas, COLUMNAS)
    guardar_csv(auditoria, inventario, COLUMNAS_AUDITORIA)
    conteo = Counter(fila["estado"] for fila in inventario)
    return {
        "procesadas": conteo["importada"] + conteo["importada_previa"],
        "descartadas": sum(n for clave, n in conteo.items()
                          if clave not in {"importada", "importada_previa", "no_seleccionada"}),
        "no_seleccionadas": conteo["no_seleccionada"],
        "pendientes": sum(fila["estado"] == "pendiente" for fila in etiquetas),
        "estados": dict(conteo),
    }


def formato_patente(patente: str) -> str:
    """Clasifica una transcripción que respeta uno de los dos formatos."""
    if not es_formato_valido(patente):
        raise ValueError(f"Formato argentino inválido: {patente!r}.")
    return "antiguo" if len(patente) == 6 else "mercosur"


def etiquetar_fila(fila: dict, texto: str, observaciones: str, filas: list[dict]) -> None:
    """Registra una transcripción humana o uno de los motivos de exclusión."""
    texto = texto.strip()
    estado = {":ilegible": "ilegible", ":fuera": "fuera_de_alcance"}.get(texto.casefold())
    if estado:
        fila.update(estado=estado, patente="", formato="", observaciones=observaciones)
        return
    if not re.fullmatch(r"[a-zA-Z0-9\s-]+", texto):
        raise ValueError("Ingresá letras ASCII y números, con espacios o guiones opcionales.")
    patente = re.sub(r"[\s-]+", "", texto).upper()
    formato = formato_patente(patente)
    grupo = [otra for otra in filas if otra is not fila and otra["patente"] == patente
             and otra["estado"] == "verificada"]
    particiones = {otra["particion"] for otra in grupo + [fila]} - {""}
    if len(particiones) > 1 or "excluida" in particiones:
        for otra in grupo + [fila]:
            otra["particion"] = "excluida"
    fila.update(estado="verificada", patente=patente, formato=formato, observaciones=observaciones)


def dividir_experimento(filas: list[dict], semilla=42) -> tuple[int, int]:
    """Fija cerca de 70/30 de imágenes agrupadas por matrícula verificada."""
    gestionadas = [fila for fila in filas if fila["id_imagen"]]
    grupos = defaultdict(list)
    if any(fila["estado"] == "pendiente" for fila in gestionadas):
        raise ValueError("Resolvé todas las etiquetas pendientes antes de fijar la división.")
    if any(fila["particion"] for fila in gestionadas):
        raise ValueError("La división experimental ya está fijada.")
    for fila in gestionadas:
        if fila["estado"] == "verificada":
            grupos[fila["patente"]].append(fila)
    claves = sorted(grupos)
    if len(claves) < 2:
        raise ValueError("Se requieren al menos dos patentes distintas verificadas.")
    random.Random(semilla).shuffle(claves)
    cantidad = sum(map(len, grupos.values()))
    # Guarda una combinación por tamaño alcanzable, siempre en grupos completos.
    combinaciones = {0: ()}
    for clave in claves:
        for tamano, elegidas in list(combinaciones.items()):
            combinaciones.setdefault(tamano + len(grupos[clave]), (*elegidas, clave))
    desarrollo = min(
        (tamano for tamano in combinaciones if 0 < tamano < cantidad),
        key=lambda tamano: abs(10 * tamano - 7 * cantidad),
    )
    seleccion = set(combinaciones[desarrollo])
    for clave in claves:
        for fila in grupos[clave]:
            fila["particion"] = "desarrollo" if clave in seleccion else "evaluacion_final"
    return desarrollo, cantidad - desarrollo
