"""Anotaciones humanas independientes y evaluación espacial sobre recortes XML."""

import csv
from pathlib import Path

from .dataset import CAJA, guardar_csv
from .evaluation import iou

CAMPOS = ("id_imagen", "indice", "caracter", "estado", *CAJA, "observacion")


def leer_anotaciones(ruta: Path) -> dict:
    if not ruta.is_file():
        return {}
    salida = {}
    with ruta.open(encoding="utf-8-sig", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            clave = fila["id_imagen"], int(fila["indice"])
            if clave in salida or fila["estado"] not in ("anotado", "omitido"):
                raise ValueError(f"Anotación individual duplicada o inválida: {clave}")
            salida[clave] = fila
    return salida


def guardar_anotaciones(ruta: Path, anotaciones: dict):
    guardar_csv(ruta, [anotaciones[k] for k in sorted(anotaciones)], CAMPOS)


def caja_anotada(fila, ancho, alto):
    """Valida coordenadas semiabiertas del recorte XML."""
    caja = tuple(int(fila[c]) for c in CAJA)
    x0, y0, x1, y1 = caja
    if not (0 <= x0 < x1 <= ancho and 0 <= y0 < y1 <= alto):
        raise ValueError(f"Caja individual fuera del recorte: {fila['id_imagen']} / {fila['indice']}")
    return caja


def estado_anotacion(muestra, anotaciones):
    presentes = [anotaciones.get((muestra["id_imagen"], i))
                 for i in range(1, len(muestra["patente"]) + 1)]
    if not any(presentes):
        return "pendiente"
    if all(f and f["estado"] == "anotado" for f in presentes):
        return "completa"
    return "parcial"


def referencias_xml(muestra, anotaciones, ancho, alto):
    """Devuelve las cajas verificadas en orden si la matrícula está completa."""
    ident = muestra["id_imagen"]
    if estado_anotacion(muestra, anotaciones) != "completa":
        return None
    salida = []
    for i, letra in enumerate(muestra["patente"], 1):
        fila = anotaciones[ident, i]
        if fila["caracter"] != letra:
            raise ValueError(f"La clase anotada discrepa de la transcripción humana: {ident} / {i}")
        salida.append(caja_anotada(fila, ancho, alto))
    return salida


def emparejar_cajas(referencias, detectadas, umbral=.5):
    """Maximiza el número de pares elegibles y luego la suma de IoU."""
    estados = {0: (0., ())}
    for j, predicha in enumerate(detectadas):
        elegibles = [(i, iou(caja, predicha)) for i, caja in enumerate(referencias)]
        nuevos = estados.copy()
        for mascara, (total, pares) in estados.items():
            for i, solape in elegibles:
                bit = 1 << i
                if mascara & bit or solape < umbral:
                    continue
                actual = mascara | bit
                if actual not in nuevos or nuevos[actual][0] < total + solape:
                    nuevos[actual] = (total + solape, (*pares, (i, j, solape)))
        estados = nuevos
    _, pares = max(estados.values(), key=lambda fila: (len(fila[1]), fila[0]))
    return sorted(pares)


def _area(caja):
    return (caja[2] - caja[0]) * (caja[3] - caja[1])


def interseccion(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def evaluar_espacial(referencias, detectadas, banda, umbral=.5):
    """Distingue aciertos uno a uno, cobertura y patrones de error geométrico."""
    pares = emparejar_cajas(referencias, detectadas, umbral)
    por_ref = {i: (j, valor) for i, j, valor in pares}
    cobertura_banda = ([interseccion(a, banda) / _area(a) for a in referencias]
                       if banda is not None else [None] * len(referencias))
    coberturas = [[interseccion(a, d) for d in detectadas] for a in referencias]
    unidos = sum(sum(coberturas[i][j] >= .25 * _area(a) for i, a in enumerate(referencias)) >= 2
                 for j in range(len(detectadas)))
    incompletos = fragmentados = 0
    detalle = []
    for i, a in enumerate(referencias):
        partes = [cruce / _area(a) for cruce, d in zip(coberturas[i], detectadas)
                  if cruce >= .10 * _area(a) and cruce >= .50 * _area(d)]
        fragmentado = len(partes) >= 2
        incompleto = len(partes) == 1 and partes[0] < .80
        fragmentados += fragmentado
        incompletos += incompleto
        indice, valor = por_ref.get(i, (None, 0.))
        detalle.append({
            "indice_anotado": i + 1, "indice_detectado": None if indice is None else indice + 1,
            "iou": valor, "cobertura_banda": cobertura_banda[i], "fragmentado": fragmentado,
            "incompleto": incompleto,
        })
    correctas = len(pares)
    return {
        "tp": correctas, "fp": len(detectadas) - correctas, "fn": len(referencias) - correctas,
        "iou_acumulado": sum(v for _, _, v in pares),
        "todos": correctas == len(referencias) == len(detectadas),
        "banda_completa": all(c >= .98 for c in cobertura_banda) if banda is not None else None,
        "incompletos": incompletos, "fragmentados": fragmentados, "unidos": unidos,
    }, detalle
