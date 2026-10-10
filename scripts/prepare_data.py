"""Preparacion del dataset: descarga, redimensionado e indice.

El corpus original son ~50 GB de JPEG con lado mayor de 500 px. Entrenar
directamente sobre el equivale a decodificar 600,000 imagenes grandes por
epoca, lo que satura la CPU y deja la A100 esperando datos. Este script genera
una vez una copia con el lado corto fijo en 256 px (suficiente para recortes de
hasta 224 px), que ocupa ~13 GB y se decodifica varias veces mas rapido. El
redimensionado es la unica transformacion irreversible del pipeline y se hace
antes de cualquier experimento para que todos vean exactamente los mismos
pixeles.

Fases:
    download  descarga los cuatro archivos desde S3 con conexiones paralelas
    extract   descomprime imagenes y anotaciones
    resize    genera la copia de lado corto fijo (paralelizada por procesos)
    index     convierte los JSON a Parquet
    verify    comprueba que el conteo de imagenes coincide con las anotaciones

Agrupaciones:
    prepare   extract + resize + index + verify. Es todo lo que NO necesita red,
              y por tanto lo unico que puede correr en un nodo de computo de
              Khipu, donde no hay salida a internet.
    all       download + prepare. Solo es ejecutable donde haya red.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from inat.config import Paths
from inat.data import build_index

BASE_URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021"
#: archivo comprimido -> ruta, relativa a `raw/`, que prueba que ya se extrajo.
#: Tenerla explicita hace que `extract` sea idempotente tambien para las
#: anotaciones, y no solo para los arboles de imagenes.
ARCHIVES = {
    "train_mini.tar.gz": "train_mini",
    "train_mini.json.tar.gz": "train_mini.json",
    "val.tar.gz": "val",
    "val.json.tar.gz": "val.json",
}

#: Numero de imagenes que debe contener cada arbol tras descomprimirlo. Son
#: cifras fijas y publicadas de iNaturalist 2021 Mini, y sirven para distinguir
#: una extraccion completa de una interrumpida: comprobar solo que la carpeta
#: existe daria por buena una extraccion cortada a la mitad, y el entrenamiento
#: seguiria adelante sobre un corpus incompleto.
EXPECTED_IMAGES = {"train_mini": 500_000, "val": 100_000}

#: Lado corto de la copia de trabajo. 256 px deja margen para RandomResizedCrop
#: a 224 px sin volver a ampliar la imagen.
SHORT_SIDE = 256
JPEG_QUALITY = 87


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------


def _remote_size(url: str) -> int:
    result = subprocess.run(
        ["curl", "-sI", "--max-time", "60", url], capture_output=True, text=True
    )
    if result.returncode != 0:
        # curl 6 = no se pudo resolver el host. En Khipu es el sintoma de estar
        # en un nodo de computo, que no tiene salida a internet.
        raise SystemExit(
            f"No se pudo contactar S3 (curl {result.returncode}). La descarga requiere "
            "red: ejecutarla en el nodo de acceso y usar la fase 'prepare' en los jobs."
        )
    out = result.stdout
    for line in out.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":")[1].strip())
    raise RuntimeError(f"No se pudo leer el tamano de {url}")


def _download_part(url: str, part: Path, start: int, end: int) -> None:
    """Descarga un rango del archivo, reanudando si la parte ya existe.

    S3 soporta peticiones por rango; varias conexiones en paralelo multiplican
    el ancho de banda efectivo frente a una sola descarga secuencial.
    """
    have = part.stat().st_size if part.exists() else 0
    if have >= end - start + 1:
        return
    subprocess.run(
        ["curl", "-sS", "--retry", "5", "--retry-delay", "5", "-r",
         f"{start + have}-{end}", "-o", str(part), "--create-dirs",
         *( ["--append"] if have else [] ), url],
        check=True,
    )


def download(paths: Paths, connections: int = 8) -> None:
    paths.raw.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVES:
        target = paths.raw / name
        url = f"{BASE_URL}/{name}"
        size = _remote_size(url)
        if target.exists() and target.stat().st_size == size:
            print(f"  {name}: ya descargado ({size / 2**30:.1f} GB)")
            continue

        print(f"  {name}: {size / 2**30:.2f} GB con {connections} conexiones")
        chunk = size // connections
        ranges = [
            (i * chunk, (size - 1) if i == connections - 1 else (i + 1) * chunk - 1)
            for i in range(connections)
        ]
        started = time.perf_counter()
        parts = [paths.raw / f"{name}.part{i}" for i in range(connections)]
        with ProcessPoolExecutor(max_workers=connections) as pool:
            futures = [
                pool.submit(_download_part, url, part, lo, hi)
                for part, (lo, hi) in zip(parts, ranges)
            ]
            for future in as_completed(futures):
                future.result()

        with open(target, "wb") as out:
            for part in parts:
                with open(part, "rb") as fh:
                    while block := fh.read(1 << 24):
                        out.write(block)
                part.unlink()
        elapsed = time.perf_counter() - started
        print(f"    listo en {elapsed / 60:.1f} min ({size / 2**20 / elapsed:.1f} MB/s)")


def _tree_is_complete(root: Path, expected: int) -> tuple[bool, int]:
    """Comprueba que un arbol de imagenes este completo, no solo presente."""
    if not root.exists():
        return False, 0
    found = sum(1 for _ in root.rglob("*.jpg"))
    return found >= expected, found


def extract(paths: Paths) -> None:
    paths.raw.mkdir(parents=True, exist_ok=True)
    for name, produced in ARCHIVES.items():
        target = paths.raw / produced
        expected = EXPECTED_IMAGES.get(produced)
        if expected is not None:
            complete, found = _tree_is_complete(target, expected)
            if complete:
                print(f"  {name}: ya extraido ({found:,} imagenes)")
                continue
            if found:
                # Reanudar sobre una extraccion incompleta: tar reescribe lo que
                # ya esta y completa lo que falta.
                print(f"  {name}: extraccion incompleta ({found:,}/{expected:,}), se reanuda")
        elif target.exists():
            print(f"  {name}: ya extraido")
            continue
        archive = paths.raw / name
        if not archive.exists():
            raise SystemExit(
                f"Falta {archive}. Ejecutar la fase 'download' en una maquina con red."
            )
        print(f"  extrayendo {name}", flush=True)
        with tarfile.open(archive) as tar:
            tar.extractall(paths.raw, filter="data")


# ---------------------------------------------------------------------------
# Redimensionado
# ---------------------------------------------------------------------------


def _resize_batch(args) -> tuple[int, int]:
    """Redimensiona un lote de imagenes. Devuelve (procesadas, fallidas)."""
    jobs, src_root, dst_root = args
    src_root, dst_root = Path(src_root), Path(dst_root)
    done = failed = 0
    for rel in jobs:
        dst = dst_root / rel
        if dst.exists():
            done += 1
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src_root / rel) as img:
                img = img.convert("RGB")
                w, h = img.size
                scale = SHORT_SIDE / min(w, h)
                if scale < 1.0:
                    img = img.resize((max(round(w * scale), SHORT_SIDE),
                                      max(round(h * scale), SHORT_SIDE)),
                                     Image.BICUBIC)
                img.save(dst, "JPEG", quality=JPEG_QUALITY, optimize=False)
            done += 1
        except Exception:
            failed += 1
    return done, failed


def resize(paths: Paths, workers: int | None = None, batch: int = 512) -> None:
    """Genera la copia de lado corto fijo, en paralelo sobre todos los nucleos."""
    workers = workers or max(1, (os.cpu_count() or 8) - 2)
    relatives: list[str] = []
    for split in ("train_mini", "val"):
        root = paths.raw / split
        if not root.exists():
            raise SystemExit(f"Falta {root}; ejecutar la fase 'extract' primero.")
        for path in root.rglob("*.jpg"):
            relatives.append(str(path.relative_to(paths.raw)).replace("\\", "/"))

    print(f"  {len(relatives):,} imagenes, {workers} procesos")
    paths.images.mkdir(parents=True, exist_ok=True)
    batches = [
        (relatives[i : i + batch], str(paths.raw), str(paths.images))
        for i in range(0, len(relatives), batch)
    ]

    done = failed = 0
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, (ok, bad) in enumerate(pool.map(_resize_batch, batches), 1):
            done += ok
            failed += bad
            if n % 50 == 0 or n == len(batches):
                rate = done / max(time.perf_counter() - started, 1e-9)
                eta = (len(relatives) - done) / max(rate, 1e-9) / 60
                print(f"    {done:,}/{len(relatives):,}  {rate:.0f} img/s  ETA {eta:.0f} min",
                      flush=True)
    size = sum(f.stat().st_size for f in paths.images.rglob("*.jpg"))
    print(f"  listo: {done:,} imagenes ({size / 2**30:.1f} GB), {failed} fallidas")


# ---------------------------------------------------------------------------
# Indice y verificacion
# ---------------------------------------------------------------------------


def index(paths: Paths, force: bool = False) -> None:
    build_index(
        paths,
        {"train": paths.raw / "train_mini.json", "val": paths.raw / "val.json"},
        force=force,
    )
    for name in ("train", "val", "categories"):
        path = paths.index / f"{name}.parquet"
        print(f"  {name}.parquet: {path.stat().st_size / 2**20:.1f} MB")


def verify(paths: Paths) -> int:
    """Comprueba que cada fila del indice tenga su imagen redimensionada."""
    import pandas as pd

    problems = 0
    for split in ("train", "val"):
        frame = pd.read_parquet(paths.index / f"{split}.parquet")
        missing = sum(1 for f in frame["file_name"] if not (paths.images / f).exists())
        print(f"  {split}: {len(frame):,} anotaciones, {missing} imagenes faltantes, "
              f"{frame['category_id'].nunique():,} especies")
        problems += missing
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["download", "extract", "resize", "index",
                                          "verify", "prepare", "all"])
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    paths = Paths(data_root=args.data_root or Paths().data_root)
    GROUPS = {
        "prepare": ["extract", "resize", "index", "verify"],
        "all": ["download", "extract", "resize", "index", "verify"],
    }
    phases = GROUPS.get(args.phase, [args.phase])

    for phase in phases:
        print(f"\n=== {phase} ===", flush=True)
        started = time.perf_counter()
        if phase == "download":
            download(paths, args.connections)
        elif phase == "extract":
            extract(paths)
        elif phase == "resize":
            resize(paths, args.workers)
        elif phase == "index":
            index(paths, args.force)
        elif phase == "verify":
            if verify(paths):
                print("  VERIFICACION FALLIDA: faltan imagenes")
                return 1
        print(f"    ({(time.perf_counter() - started) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
