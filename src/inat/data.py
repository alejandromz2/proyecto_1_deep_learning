"""Pipeline de datos: anotaciones -> indice -> Dataset -> DataLoader.

El dataset original se distribuye como JSON estilo COCO (`images`,
`annotations`, `categories`) mas un arbol de carpetas por especie. Leer esos
JSON cuesta varios segundos y no cambia entre corridas, asi que se convierten
una sola vez a tablas Parquet (`build_index`) que el resto del proyecto
consume. Las imagenes se leen desde la copia redimensionada que genera
`scripts/prepare_data.py`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from .config import DataConfig, Paths
from .seeding import make_generator, worker_init_fn

# Estadisticas de ImageNet. Se usan incluso al entrenar desde cero para que la
# normalizacion no sea una variable adicional entre experimentos.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]


# ---------------------------------------------------------------------------
# Construccion del indice
# ---------------------------------------------------------------------------


def build_index(paths: Paths, annotations: dict, force: bool = False) -> None:
    """Convierte los JSON de anotaciones en tablas Parquet.

    `annotations` mapea el nombre del split ("train", "val") al JSON
    correspondiente. Se escriben `<split>.parquet` con una fila por imagen y
    `categories.parquet` con la taxonomia de las 10,000 especies.
    """
    paths.index.mkdir(parents=True, exist_ok=True)
    categories_written = (paths.index / "categories.parquet").exists() and not force

    for split, json_path in annotations.items():
        out = paths.index / f"{split}.parquet"
        if out.exists() and not force:
            continue
        with open(json_path, encoding="utf-8") as fh:
            blob = json.load(fh)

        # image_id -> category_id. Los dos arrays vienen alineados en el
        # dataset oficial, pero no se asume: se unen por clave explicita.
        label_of = {a["image_id"]: a["category_id"] for a in blob["annotations"]}
        frame = pd.DataFrame(
            {
                "file_name": [im["file_name"] for im in blob["images"]],
                "category_id": [label_of[im["id"]] for im in blob["images"]],
                "width": [im["width"] for im in blob["images"]],
                "height": [im["height"] for im in blob["images"]],
            }
        )
        frame.to_parquet(out, index=False)

        if not categories_written:
            cats = pd.DataFrame(blob["categories"])
            keep = ["id", "name", "common_name", "image_dir_name"] + RANKS
            cats = cats[[c for c in keep if c in cats.columns]]
            cats = cats.rename(columns={"id": "category_id"}).sort_values("category_id")
            cats.to_parquet(paths.index / "categories.parquet", index=False)
            categories_written = True


def load_index(paths: Paths, split: str) -> pd.DataFrame:
    return pd.read_parquet(paths.index / f"{split}.parquet")


def load_categories(paths: Paths) -> pd.DataFrame:
    return pd.read_parquet(paths.index / "categories.parquet")


# ---------------------------------------------------------------------------
# Escenarios de datos
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """Resultado de aplicar un escenario: tablas filtradas y metadatos."""

    train: pd.DataFrame
    val: pd.DataFrame
    categories: pd.DataFrame
    num_classes: int
    #: frecuencia de entrenamiento por indice de clase contiguo
    class_counts: np.ndarray
    #: "head"/"mid"/"tail" por indice de clase; None en escenario balanceado
    group_of_class: np.ndarray | None = None


def _remap_labels(train: pd.DataFrame, val: pd.DataFrame, categories: pd.DataFrame):
    """Reindexa `category_id` a etiquetas contiguas 0..C-1.

    Necesario porque los escenarios pueden quedarse con un subconjunto de
    especies y la capa de salida debe tener exactamente C neuronas.
    """
    kept = np.sort(train["category_id"].unique())
    mapping = {int(c): i for i, c in enumerate(kept)}
    train = train.assign(label=train["category_id"].map(mapping).astype("int64"))
    val = val[val["category_id"].isin(mapping)].copy()
    val["label"] = val["category_id"].map(mapping).astype("int64")
    categories = categories[categories["category_id"].isin(mapping)].copy()
    categories["label"] = categories["category_id"].map(mapping).astype("int64")
    categories = categories.sort_values("label").reset_index(drop=True)
    return train.reset_index(drop=True), val.reset_index(drop=True), categories, len(kept)


def make_scenario(paths: Paths, cfg: DataConfig) -> Scenario:
    """Construye el escenario de datos descrito por `cfg`, de forma reproducible.

    Todo el muestreo usa un `numpy.random.Generator` sembrado con
    `cfg.longtail_seed`; no hay seleccion manual de archivos en ninguna etapa.
    """
    train = load_index(paths, "train")
    val = load_index(paths, "val")
    categories = load_categories(paths)
    rng = np.random.default_rng(cfg.longtail_seed)

    # Submuestreo opcional de especies (para depuracion y corridas piloto).
    if cfg.subset_classes is not None:
        all_cats = np.sort(train["category_id"].unique())
        n = min(cfg.subset_classes, len(all_cats))
        chosen = np.sort(rng.choice(all_cats, size=n, replace=False))
        train = train[train["category_id"].isin(chosen)]
        val = val[val["category_id"].isin(chosen)]

    if cfg.subset_images_per_class is not None:
        quota = {int(c): cfg.subset_images_per_class for c in train["category_id"].unique()}
        train = _sample_per_class(train, quota, rng)

    group_series = None
    if cfg.scenario == "longtail":
        train, group_series = _build_longtail(train, cfg, rng)

    train, val, categories, num_classes = _remap_labels(train, val, categories)

    group_of_class = None
    if group_series is not None:
        # El grupo se asigno por category_id; se traduce al indice contiguo.
        lookup = dict(zip(categories["category_id"], categories["label"]))
        group_of_class = np.empty(num_classes, dtype=object)
        for cat_id, grp in group_series.items():
            if cat_id in lookup:
                group_of_class[lookup[cat_id]] = grp

    counts = np.bincount(train["label"].to_numpy(), minlength=num_classes)
    return Scenario(train, val, categories, num_classes, counts, group_of_class)


def _sample_per_class(frame: pd.DataFrame, quota: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Toma hasta `quota[c]` filas de cada clase `c` sin reemplazo."""
    pieces = []
    for cat_id, block in frame.groupby("category_id", sort=True):
        n = min(int(quota.get(int(cat_id), len(block))), len(block))
        idx = np.sort(rng.choice(len(block), size=n, replace=False))
        pieces.append(block.iloc[idx])
    return pd.concat(pieces, ignore_index=True)


def _build_longtail(train: pd.DataFrame, cfg: DataConfig, rng: np.random.Generator):
    """Construye la distribucion long-tail por codigo.

    Criterio: las especies se barajan con la semilla declarada y se parten en
    tres grupos disjuntos -- frecuentes, intermedias y minoritarias -- a los que
    se asigna una cuota fija de imagenes de entrenamiento. Se barajan en lugar
    de ordenarlas por algun atributo para que la pertenencia al grupo sea
    independiente de la dificultad taxonomica: asi la caida de rendimiento en
    la cola se atribuye al numero de ejemplos y no al tipo de organismo.
    """
    cats = np.sort(train["category_id"].unique())
    shuffled = rng.permutation(cats)
    n_head = int(round(cfg.longtail_head_frac * len(cats)))
    n_mid = int(round(cfg.longtail_mid_frac * len(cats)))
    groups: dict[int, str] = {}
    quota: dict[int, int] = {}
    for i, cat_id in enumerate(shuffled):
        if i < n_head:
            groups[int(cat_id)], quota[int(cat_id)] = "head", cfg.longtail_head_n
        elif i < n_head + n_mid:
            groups[int(cat_id)], quota[int(cat_id)] = "mid", cfg.longtail_mid_n
        else:
            groups[int(cat_id)], quota[int(cat_id)] = "tail", cfg.longtail_tail_n
    return _sample_per_class(train, quota, rng), pd.Series(groups)


# ---------------------------------------------------------------------------
# Dataset y transformaciones
# ---------------------------------------------------------------------------


class INatDataset(Dataset):
    """Lee imagenes JPEG ya redimensionadas y devuelve (tensor, etiqueta)."""

    def __init__(self, frame: pd.DataFrame, root: Path, transform):
        self.files = frame["file_name"].to_numpy()
        self.labels = frame["label"].to_numpy()
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.root / self.files[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.transform(img), int(self.labels[idx])


def build_transforms(cfg: DataConfig, train: bool):
    """Pipeline de transformaciones.

    En validacion se usa resize + center crop determinista: la evaluacion no
    debe introducir varianza propia al comparar experimentos.
    """
    size = cfg.image_size
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    if not train:
        return transforms.Compose(
            [
                transforms.Resize(int(size * 1.14)),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ]
        )

    ops = []
    if cfg.random_resized_crop:
        ops.append(transforms.RandomResizedCrop(size, scale=(cfg.rrc_scale_min, 1.0)))
    else:
        ops += [transforms.Resize(int(size * 1.14)), transforms.CenterCrop(size)]
    if cfg.horizontal_flip:
        ops.append(transforms.RandomHorizontalFlip())
    if cfg.rotation_degrees > 0:
        ops.append(transforms.RandomRotation(cfg.rotation_degrees))
    if cfg.color_jitter > 0:
        j = cfg.color_jitter
        ops.append(
            transforms.ColorJitter(brightness=j, contrast=j, saturation=j, hue=min(j / 2, 0.5))
        )
    ops += [transforms.ToTensor(), normalize]
    if cfg.random_erasing > 0:
        ops.append(transforms.RandomErasing(p=cfg.random_erasing, value="random"))
    return transforms.Compose(ops)


def build_loaders(
    scenario: Scenario,
    cfg: DataConfig,
    paths: Paths,
    seed: int,
    weighted_sampler: bool = False,
):
    """Crea los DataLoaders de entrenamiento y validacion.

    El sampler balanceado es una de las estrategias de mitigacion del
    desbalance; cuando esta activo se desactiva `shuffle` porque el propio
    sampler define el orden.
    """
    train_ds = INatDataset(scenario.train, paths.images, build_transforms(cfg, train=True))
    val_ds = INatDataset(scenario.val, paths.images, build_transforms(cfg, train=False))

    sampler = None
    shuffle = True
    if weighted_sampler:
        per_class = 1.0 / np.maximum(scenario.class_counts, 1)
        weights = per_class[scenario.train["label"].to_numpy()]
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
            generator=make_generator(seed),
        )
        shuffle = False

    common = dict(
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )
    if cfg.num_workers > 0:
        common["prefetch_factor"] = cfg.prefetch_factor

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=True,
        generator=make_generator(seed),
        **common,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.eval_batch_size, shuffle=False, drop_last=False, **common
    )
    return train_loader, val_loader
