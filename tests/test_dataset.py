import sys
from xml.etree import ElementTree as ET

import pytest
from PIL import Image

from scripts import label_dataset
from src import dataset
from src.dataset import (
    COLUMNAS, dividir_experimento, etiquetar_fila, guardar_csv, leer_caja_voc,
    leer_etiquetas, preparar_dataset,
)


def _anotacion(ruta, dimensiones=(12, 8), cajas=((2, 1, 9, 6),)):
    raiz = ET.Element("annotation")
    ET.SubElement(raiz, "filename").text = ruta.name
    tamano = ET.SubElement(raiz, "size")
    for lado, valor in zip(("width", "height"), dimensiones):
        ET.SubElement(tamano, lado).text = str(valor)
    for caja in cajas:
        objeto = ET.SubElement(raiz, "object")
        ET.SubElement(objeto, "name").text = "patente"
        box = ET.SubElement(objeto, "bndbox")
        for coordenada, valor in zip(("xmin", "ymin", "xmax", "ymax"), caja):
            ET.SubElement(box, coordenada).text = str(valor)
    xml = ruta.with_suffix(".xml")
    ET.ElementTree(raiz).write(xml, encoding="utf-8")
    return xml


@pytest.mark.parametrize(
    ("coordenadas", "caja", "esperada"),
    [("roboflow", (0, 0, 12, 8), (0, 0, 12, 8)), ("voc", (1, 2, 12, 8), (0, 1, 12, 8))],
)
def test_interpretar_voc(tmp_path, coordenadas, caja, esperada):
    imagen = tmp_path / "foto.jpg"
    xml = _anotacion(imagen, cajas=(caja,))
    assert leer_caja_voc(xml, imagen.name, (12, 8), coordenadas) == esperada
    with pytest.raises(ValueError, match="Dimensiones"):
        leer_caja_voc(xml, imagen.name, (12, 7), coordenadas)
    with pytest.raises(ValueError, match="corresponde"):
        leer_caja_voc(xml, "otra.jpg", (12, 8), coordenadas)


@pytest.mark.parametrize("caja", [(-1, 1, 5, 6), (2, 1, 13, 6), (2, 1, 2, 6), (2, 4, 5, 2)])
def test_rechazar_cajas_invalidas(tmp_path, caja):
    imagen = tmp_path / "foto.jpg"
    xml = _anotacion(imagen, cajas=(caja,))
    with pytest.raises(ValueError, match="Caja inválida"):
        leer_caja_voc(xml, imagen.name, (12, 8))


def test_importar_voc_preserva_originales_y_registra_descartes(tmp_path):
    fuente = tmp_path / "raw"
    train = fuente / "train"
    train.mkdir(parents=True)
    foto = Image.new("RGB", (12, 8))
    foto.putpixel((2, 1), (9, 8, 7))
    for nombre in ("a", "b_duplicada", "sin_etiqueta", "multiples"):
        foto.save(train / f"{nombre}.png")
    _anotacion(train / "a.png")
    _anotacion(train / "b_duplicada.png")
    _anotacion(train / "multiples.png", cajas=((2, 1, 9, 6), (1, 1, 8, 7)))
    (train / "corrupta.jpg").write_bytes(b"no es una imagen")
    _anotacion(train / "corrupta.jpg")
    _anotacion(train / "huerfana.jpg")
    (train / "archivo.invalido").touch()
    huellas = {p: p.read_bytes() for p in train.iterdir()}

    datos = tmp_path / "data"
    log = tmp_path / "log.csv"
    resumen = preparar_dataset(fuente, datos, log, limite=1, semilla=12, fuente="Fixture")
    assert resumen["procesadas"] == resumen["pendientes"] == 1
    assert resumen["estados"] == {
        "importada": 1, "duplicada": 1, "imagen_invalida": 1, "sin_xml": 1,
        "anotacion_invalida": 1, "xml_sin_imagen": 1, "archivo_invalido": 1,
    }
    assert all(p.read_bytes() == contenido for p, contenido in huellas.items())
    fila, = leer_etiquetas(datos / "ground_truth.csv")
    assert (datos / fila["original"]).read_bytes() == huellas[train / "a.png"]
    with Image.open(datos / fila["imagen"]) as recorte:
        assert recorte.size == (7, 5)
        assert recorte.getpixel((0, 0)) == (9, 8, 7)
    assert fila["particion_origen"] == "train" and fila["procedencia"] == "Fixture"
    assert fila["estado"] == "pendiente" and not fila["patente"]

    etiquetar_fila(fila, "abc-123", "Revisada", [fila])
    guardar_csv(datos / "ground_truth.csv", [fila], COLUMNAS)
    preparar_dataset(fuente, datos, log, limite=1, semilla=12, fuente="Fixture")
    assert leer_etiquetas(datos / "ground_truth.csv")[0]["patente"] == "ABC123"


def test_importacion_permite_reintentar_una_copia_interrumpida(tmp_path, monkeypatch):
    fuente = tmp_path / "raw"
    fuente.mkdir()
    original = fuente / "foto.png"
    Image.new("RGB", (12, 8)).save(original)
    _anotacion(original)
    datos = tmp_path / "data"
    log = tmp_path / "log.csv"

    def interrumpir(origen, destino):
        destino.write(origen.read(8))
        raise OSError("Escritura interrumpida")

    with monkeypatch.context() as parche:
        parche.setattr(dataset.shutil, "copyfileobj", interrumpir)
        with pytest.raises(OSError, match="interrumpida"):
            preparar_dataset(fuente, datos, log)
    assert not list((datos / "original").iterdir())
    assert not (datos / "ground_truth.csv").exists()
    assert preparar_dataset(fuente, datos, log)["procesadas"] == 1
    fila, = leer_etiquetas(datos / "ground_truth.csv")
    assert (datos / fila["original"]).read_bytes() == original.read_bytes()


def test_csv_anterior_y_division_sin_fuga(tmp_path):
    ruta = tmp_path / "ground_truth.csv"
    ruta.write_text("imagen,patente\ncropped/legado.png,ABC999\n", encoding="utf-8")
    legado, = leer_etiquetas(ruta)
    assert legado["patente"] == "ABC999" and legado["particion"] == ""

    filas = [
        dict.fromkeys(COLUMNAS, "") | {"id_imagen": str(i), "estado": "pendiente"}
        for i in range(10)
    ]
    with pytest.raises(ValueError, match="pendientes"):
        dividir_experimento(filas)
    for i, fila in enumerate(filas):
        etiquetar_fila(fila, f"ab{(i // 2 if i < 4 else i):03}cd", "", filas)
    assert dividir_experimento(filas, semilla=42) == (7, 3)
    assert len({f["particion"] for f in filas if f["patente"] == "AB000CD"}) == 1
    assert len({f["particion"] for f in filas if f["patente"] == "AB001CD"}) == 1
    with pytest.raises(ValueError, match="ya está fijada"):
        dividir_experimento(filas)
    desarrollo = next(f for f in filas if f["particion"] == "desarrollo")
    reservada = next(f for f in filas if f["particion"] == "evaluacion_final")
    etiquetar_fila(desarrollo, reservada["patente"], "", filas)
    assert desarrollo["particion"] == reservada["particion"] == "excluida"


def test_estados_manuales_y_validacion_de_transcripcion():
    fila = dict.fromkeys(COLUMNAS, "")
    for invalida in ("ABC12S", "AB123CÑ", "AB123CD?"):
        with pytest.raises(ValueError):
            etiquetar_fila(fila, invalida, "", [fila])
    etiquetar_fila(fila, ":ilegible", "Reflejo", [fila])
    assert fila["estado"] == "ilegible" and fila["observaciones"] == "Reflejo"
    etiquetar_fila(fila, ":fuera", "Motocicleta", [fila])
    assert fila["estado"] == "fuera_de_alcance"
    etiquetar_fila(fila, "ab 123-cd", "Corregida", [fila])
    assert (fila["estado"], fila["patente"], fila["formato"]) == ("verificada", "AB123CD", "mercosur")


def test_cli_etiquetado_guarda_reanuda_y_corrige(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ruta = tmp_path / "data/ground_truth.csv"
    filas = [
        {"imagen": f"cropped/{i}.png", "patente": "", "id_imagen": str(i), "estado": "pendiente"}
        for i in range(3)
    ]
    guardar_csv(ruta, filas, COLUMNAS)
    mostradas = []
    monkeypatch.setattr(label_dataset, "_mostrar", lambda _, fila: mostradas.append(fila["id_imagen"]))

    def ejecutar(argumentos, respuestas=()):
        entradas = iter(respuestas)
        monkeypatch.setattr(sys, "argv", ["label_dataset", *argumentos])
        monkeypatch.setattr("builtins.input", lambda _: next(entradas))
        label_dataset.main()

    ejecutar([], ["abc123", "Legible", ":salir"])
    assert [fila["estado"] for fila in leer_etiquetas(ruta)] == ["verificada", "pendiente", "pendiente"]
    assert mostradas == ["0", "1"]
    mostradas.clear()
    ejecutar([], ["AB123CD", "", ":ilegible", "Reflejo"])
    assert mostradas == ["1", "2"]
    ejecutar(["--corregir", "0"], ["ABC124", "Lectura corregida"])
    ejecutar(["--dividir", "--semilla", "42"])
    filas = leer_etiquetas(ruta)
    assert [fila["patente"] for fila in filas] == ["ABC124", "AB123CD", ""]
    assert {fila["particion"] for fila in filas[:2]} == {"desarrollo", "evaluacion_final"}
