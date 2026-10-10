"""Comprobaciones de solo lectura antes de enviar los jobs.

Se ejecuta en el nodo de acceso: valida sintaxis, entorno, datos y cola sin
reservar GPU ni entrenar nada.
"""
import ast
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

problems = []

for path in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py")):
    try:
        ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        problems.append(f"sintaxis en {path.relative_to(ROOT)}: {exc}")

import torch, torchvision
print(f"torch {torch.__version__} / torchvision {torchvision.__version__}")

os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch"))
cache = Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints"
weights = list(cache.glob("*.pth")) if cache.exists() else []
print(f"pesos de ImageNet en cache: {len(weights)}")
if not weights:
    problems.append("no hay pesos preentrenados en cache; los nodos de computo no tienen red")

data_root = Path(os.environ.get("INAT_DATA_ROOT", Path.home() / "datasets/inat2021"))
for name in ("train", "val", "categories"):
    f = data_root / "index" / f"{name}.parquet"
    print(f"{f}: {'ok' if f.exists() else 'FALTA'}")
    if not f.exists():
        problems.append(f"falta {f}")

if (data_root / "index/train.parquet").exists():
    import pandas as pd
    tr = pd.read_parquet(data_root / "index/train.parquet")
    va = pd.read_parquet(data_root / "index/val.parquet")
    print(f"train: {len(tr):,} imagenes / {tr.category_id.nunique():,} especies")
    print(f"val:   {len(va):,} imagenes / {va.category_id.nunique():,} especies")
    missing = sum(1 for f in tr.file_name.head(200) if not (data_root / "images" / f).exists())
    print(f"muestra de 200 imagenes: {missing} faltantes")
    if missing:
        problems.append("faltan imagenes redimensionadas")

from inat.registry import PHASES
total = sum(len(build()) for build in PHASES.values())
print(f"cola declarada: {total} experimentos en {len(PHASES)} fases")

if problems:
    print("\nPREFLIGHT FALLIDO:")
    for p in problems:
        print(" -", p)
    sys.exit(1)
print("\nPREFLIGHT OK")
