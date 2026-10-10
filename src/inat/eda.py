"""Analisis exploratorio del corpus.

Produce las cifras y las selecciones de imagenes que sostienen la discusion de
por que este problema es dificil: cuantas especies hay, como se reparten por
grupo taxonomico, cuanto varia una misma especie entre fotografias, y cuanto se
parecen entre si especies distintas del mismo genero.

La seleccion de ejemplos visuales es aleatoria con semilla fija, no manual: de
otro modo el informe mostraria los casos que confirman la tesis.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import Paths
from .data import RANKS, load_categories, load_index


@dataclass
class CorpusSummary:
    """Cifras basicas del corpus, para la tabla de la Parte 1."""

    n_train: int
    n_val: int
    n_species: int
    images_per_species_train: pd.Series
    images_per_species_val: pd.Series
    rank_counts: dict
    dimensions: pd.DataFrame

    def as_frame(self) -> pd.DataFrame:
        rows = [
            ("Imagenes de entrenamiento", f"{self.n_train:,}"),
            ("Imagenes de validacion", f"{self.n_val:,}"),
            ("Especies", f"{self.n_species:,}"),
            ("Imagenes por especie (train)",
             f"{self.images_per_species_train.min()}-{self.images_per_species_train.max()}"),
            ("Imagenes por especie (val)",
             f"{self.images_per_species_val.min()}-{self.images_per_species_val.max()}"),
        ]
        for rank in RANKS:
            if rank in self.rank_counts:
                rows.append((f"Taxones distintos - {rank}", f"{self.rank_counts[rank]:,}"))
        rows += [
            ("Ancho mediano (px)", f"{self.dimensions['width'].median():.0f}"),
            ("Alto mediano (px)", f"{self.dimensions['height'].median():.0f}"),
            ("Lado mayor maximo (px)",
             f"{self.dimensions[['width', 'height']].max().max():.0f}"),
        ]
        return pd.DataFrame(rows, columns=["Caracteristica", "Valor"])


def summarize(paths: Paths, sample_dimensions: int = 50_000,
              seed: int = 42) -> CorpusSummary:
    """Resumen cuantitativo del corpus completo."""
    train = load_index(paths, "train")
    val = load_index(paths, "val")
    cats = load_categories(paths)
    rng = np.random.default_rng(seed)

    sample = train
    if len(train) > sample_dimensions:
        idx = rng.choice(len(train), size=sample_dimensions, replace=False)
        sample = train.iloc[np.sort(idx)]

    return CorpusSummary(
        n_train=len(train), n_val=len(val),
        # Especies efectivamente presentes en entrenamiento, no las declaradas
        # en la taxonomia: en el corpus completo coinciden, y en un subset de
        # desarrollo esta es la cifra que describe lo que el modelo ve.
        n_species=train["category_id"].nunique(),
        images_per_species_train=train.groupby("category_id").size(),
        images_per_species_val=val.groupby("category_id").size(),
        rank_counts={r: cats[r].nunique() for r in RANKS if r in cats.columns},
        dimensions=sample[["width", "height"]].reset_index(drop=True),
    )


def species_per_rank(paths: Paths, rank: str = "class") -> pd.Series:
    """Numero de especies por taxon de un rango dado (p. ej. clase: Aves, Insecta)."""
    cats = load_categories(paths)
    if rank not in cats.columns:
        raise ValueError(f"rango desconocido: {rank}")
    return cats.groupby(rank).size().sort_values(ascending=False)


def images_per_rank(paths: Paths, split: str = "train", rank: str = "class") -> pd.Series:
    """Numero de imagenes por taxon de un rango dado."""
    frame = load_index(paths, split)
    cats = load_categories(paths)[["category_id", rank]]
    merged = frame.merge(cats, on="category_id", how="left")
    return merged.groupby(rank).size().sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Ejemplos visuales
# ---------------------------------------------------------------------------


def _load(paths: Paths, file_name: str, size: int = 224) -> Image.Image:
    with Image.open(paths.images / file_name) as img:
        img = img.convert("RGB")
        return img.resize((size, size), Image.BICUBIC)


def sample_species_grid(paths: Paths, n_species: int = 10, seed: int = 42, size: int = 224):
    """Una imagen de cada una de `n_species` especies elegidas al azar."""
    train = load_index(paths, "train")
    cats = load_categories(paths).set_index("category_id")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.sort(train["category_id"].unique()), size=n_species, replace=False)

    images, captions = [], []
    for cat_id in chosen:
        block = train[train["category_id"] == cat_id]
        row = block.iloc[rng.integers(len(block))]
        images.append(_load(paths, row["file_name"], size))
        info = cats.loc[cat_id]
        captions.append(f"{info['name']}\n({info.get('class', '?')})")
    return images, captions


def intra_class_grid(paths: Paths, n: int = 8, seed: int = 42, size: int = 224,
                     category_id: int | None = None):
    """Varias fotografias de una misma especie.

    Hace visible la variabilidad intra-clase: pose, iluminacion, escala del
    organismo, fondo y estadio de desarrollo cambian mucho mas entre dos fotos
    de la misma especie que entre especies proximas.
    """
    train = load_index(paths, "train")
    cats = load_categories(paths).set_index("category_id")
    rng = np.random.default_rng(seed)
    if category_id is None:
        category_id = int(rng.choice(np.sort(train["category_id"].unique())))

    block = train[train["category_id"] == category_id]
    idx = rng.choice(len(block), size=min(n, len(block)), replace=False)
    images = [_load(paths, block.iloc[i]["file_name"], size) for i in idx]
    info = cats.loc[category_id]
    captions = [f"{block.iloc[i]['width']}x{block.iloc[i]['height']} px" for i in idx]
    title = f"Variabilidad dentro de una especie: {info['name']}"
    return images, captions, title


def inter_class_grid(paths: Paths, rank: str = "genus", n_species: int = 6,
                     seed: int = 42, size: int = 224):
    """Especies distintas del mismo genero.

    Es el caso que define el problema fine-grained: la diferencia relevante
    entre dos de estas imagenes puede ser el patron de una franja o la forma de
    un borde, mientras el fondo y la pose varian mucho mas.
    """
    train = load_index(paths, "train")
    cats = load_categories(paths)
    rng = np.random.default_rng(seed)

    # Solo cuentan las especies con imagenes en entrenamiento: la taxonomia
    # declara 10,000 especies, pero un subset de desarrollo tiene muchas menos y
    # el taxon elegido debe ser uno que realmente se pueda ilustrar.
    presentes = cats[cats["category_id"].isin(train["category_id"].unique())]
    sizes = presentes.groupby(rank).size()
    candidates = sizes[sizes >= n_species].index.to_numpy()
    if len(candidates) == 0:
        # Relajar el requisito antes de rendirse: con dos especies del mismo
        # taxon la comparacion ya es informativa.
        candidates = sizes[sizes >= 2].index.to_numpy()
        if len(candidates) == 0:
            raise ValueError(
                f"no hay ningun {rank} con al menos dos especies presentes en entrenamiento"
            )
        n_species = int(sizes[candidates].max())
    taxon = candidates[rng.integers(len(candidates))]

    members = presentes[presentes[rank] == taxon].head(n_species)
    images, captions = [], []
    for _, info in members.iterrows():
        block = train[train["category_id"] == info["category_id"]]
        if block.empty:
            continue
        row = block.iloc[rng.integers(len(block))]
        images.append(_load(paths, row["file_name"], size))
        captions.append(info["name"].split()[-1])
    rank_label = {"genus": "genero", "family": "familia", "order": "orden",
                  "class": "clase"}.get(rank, rank)
    return images, captions, f"Especies de {rank_label} {taxon}"


def difficulty_notes(summary: CorpusSummary, paths: Paths) -> pd.DataFrame:
    """Dificultades del problema, cada una ligada a una cifra del corpus.

    Sirve para que la discusion de la Parte 1 se apoye en los datos y no en
    afirmaciones generales sobre clasificacion fine-grained.
    """
    cats = load_categories(paths)
    per_genus = cats.groupby("genus").size() if "genus" in cats.columns else pd.Series(dtype=int)
    congeneric = int((per_genus > 1).sum())
    species_in_shared_genus = int(per_genus[per_genus > 1].sum())

    rows = [
        ("Numero de clases",
         f"{summary.n_species:,} especies",
         "La capa de salida domina el conteo de parametros y el azar da "
         f"{100 / summary.n_species:.2f}% de Top-1; el margen entre clases proximas es minimo."),
        ("Ejemplos por clase",
         f"{summary.images_per_species_train.median():.0f} imagenes de entrenamiento",
         "Muy pocas para aprender desde cero un concepto visual fino: es el argumento "
         "central a favor del transfer learning."),
        ("Similitud entre clases",
         f"{congeneric:,} generos con mas de una especie "
         f"({species_in_shared_genus:,} especies afectadas)",
         "Especies del mismo genero comparten morfologia; la senal discriminante es local "
         "y de baja amplitud frente a la variacion de fondo."),
        ("Variabilidad dentro de la clase",
         "pose, iluminacion, escala y fondo libres",
         "Son fotografias de ciencia ciudadana, no tomas controladas: la varianza intra-clase "
         "puede superar a la inter-clase."),
        ("Resolucion",
         f"mediana {summary.dimensions['width'].median():.0f}x"
         f"{summary.dimensions['height'].median():.0f} px",
         "Al reescalar a 160-224 px, los rasgos diagnosticos finos ocupan pocos pixeles."),
        ("Desbalance taxonomico",
         f"{len(summary.rank_counts)} rangos anotados",
         "Los grupos taxonomicos no estan igualmente representados ni son igual de dificiles; "
         "el Top-1 agregado promedia regimenes muy distintos."),
    ]
    return pd.DataFrame(rows, columns=["Dificultad", "Evidencia en el corpus", "Consecuencia"])
