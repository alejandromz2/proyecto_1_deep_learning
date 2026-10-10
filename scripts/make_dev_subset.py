"""Construye un indice de desarrollo a partir de las imagenes disponibles en disco.

El corpus completo vive en Khipu. Para poder abrir el notebook, depurar el
pipeline y verificar que los modulos funcionan sin descargar 50 GB, este script
genera un indice restringido a las imagenes que si estan presentes localmente
(el subset de 20 especies del laboratorio S03). Las tablas resultantes tienen
exactamente el mismo esquema que las del corpus completo, de modo que el codigo
de entrenamiento no distingue entre ambos entornos.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from inat.config import Paths
from inat.data import build_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="carpeta con train_mini.json, val.json y los arboles de imagenes")
    parser.add_argument("--out", type=Path, required=True,
                        help="raiz del indice de desarrollo que se va a generar")
    args = parser.parse_args()

    paths = Paths(data_root=args.out)
    build_index(paths, {"train": args.source / "train_mini.json",
                        "val": args.source / "val.json"}, force=True)

    # Filtrar a lo que existe en disco. El indice completo describe 10,000
    # especies; aqui solo hay unas pocas.
    for split in ("train", "val"):
        path = paths.index / f"{split}.parquet"
        frame = pd.read_parquet(path)
        present = frame["file_name"].map(lambda f: (args.source / f).exists())
        frame = frame[present].reset_index(drop=True)
        frame.to_parquet(path, index=False)
        print(f"{split}: {len(frame):,} imagenes, {frame['category_id'].nunique()} especies")

    print(f"\nIndice de desarrollo en {paths.index}")
    print(f"Usar INAT_DATA_ROOT={args.out} con las imagenes en {args.source}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
