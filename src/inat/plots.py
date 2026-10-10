"""Figuras del informe.

Una sola paleta categorica para todo el proyecto, verificada para daltonismo,
y etiquetas directas en lugar de leyendas cuando hay pocas series. Cada funcion
devuelve la figura para que el notebook decida si mostrarla o guardarla.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#: Paleta categorica validada (separacion CVD y contraste comprobados). El
#: orden es fijo: la serie n-esima usa siempre el mismo color, de modo que un
#: mismo experimento conserve su color entre figuras.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948"]
#: Para nubes de puntos (donde todos los pares de colores compiten a la vez)
#: solo los tres primeros mantienen la separacion garantizada.
SERIES_SCATTER = SERIES[:3]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dedcd6"

PALETTE_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def use_style() -> None:
    """Estilo comun: marcas finas, rejilla discreta, texto en tinta neutra."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.bbox": "tight", "savefig.dpi": 150,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_SOFT, "axes.titlecolor": INK,
        "axes.grid": True, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT, "text.color": INK,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 6,
        "figure.titlesize": 12, "figure.titleweight": "bold",
    })


def _finish(fig, path: Path | None):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    return fig


def _label_last(ax, x, y, text, color, dx=0.15):
    """Etiqueta directa al final de una curva; evita leyendas cuando hay pocas."""
    ax.annotate(text, xy=(x, y), xytext=(dx, 0), textcoords="offset fontsize",
                color=color, fontsize=9, va="center", fontweight="bold")


# ---------------------------------------------------------------------------
# Exploracion del dataset
# ---------------------------------------------------------------------------


def plot_taxonomic_distribution(counts: pd.Series, title: str, path=None):
    """Numero de especies por grupo taxonomico."""
    use_style()
    counts = counts.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(counts))))
    ax.barh(counts.index, counts.values, color=SERIES[0], height=0.68)
    for name, value in counts.items():
        ax.annotate(f"{value:,}", (value, name), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK_SOFT)
    ax.set_xlabel("Especies")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.12)
    return _finish(fig, path)


def plot_image_dimensions(frame: pd.DataFrame, path=None):
    """Distribucion de dimensiones y relacion de aspecto del corpus."""
    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].hist(frame["width"], bins=60, color=SERIES[0])
    axes[0].set_title("Ancho (px)")
    axes[1].hist(frame["height"], bins=60, color=SERIES[1])
    axes[1].set_title("Alto (px)")
    aspect = frame["width"] / frame["height"]
    axes[2].hist(aspect, bins=60, color=SERIES[2])
    axes[2].axvline(1.0, color=INK_SOFT, linewidth=1, linestyle="--")
    axes[2].set_title("Relacion de aspecto")
    for ax in axes:
        ax.set_ylabel("Imagenes")
    fig.suptitle("Dimensiones de las imagenes")
    fig.tight_layout()
    return _finish(fig, path)


def plot_image_grid(images, titles, ncols=5, title="", path=None, suptitle_size=12):
    """Rejilla de imagenes con pie de foto; se usa en EDA y analisis de errores."""
    use_style()
    n = len(images)
    if n == 0:
        fig, ax = plt.subplots(figsize=(5, 1.4))
        ax.text(0.5, 0.5, "sin imagenes disponibles", ha="center", va="center",
                color=INK_SOFT)
        ax.axis("off")
        return _finish(fig, path)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.55 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, img, caption in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(caption, fontsize=8, color=INK_SOFT)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=suptitle_size)
    fig.tight_layout()
    return _finish(fig, path)


# ---------------------------------------------------------------------------
# Curvas de aprendizaje
# ---------------------------------------------------------------------------


def plot_learning_curves(history: pd.DataFrame, labels: dict | None = None,
                         title="", path=None):
    """Cuatro paneles: perdida y exactitud, en entrenamiento y validacion.

    Es la figura que el enunciado pide para comparar optimizadores, y sirve
    igual para arquitecturas o estrategias de fine-tuning.
    """
    use_style()
    runs = list(dict.fromkeys(history["run_name"]))
    labels = labels or {r: r for r in runs}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = [("train_loss", "Perdida - entrenamiento"), ("val_loss", "Perdida - validacion"),
              ("train_top1", "Top-1 - entrenamiento"), ("val_top1", "Top-1 - validacion")]

    for ax, (column, panel_title) in zip(axes.ravel(), panels):
        for i, run in enumerate(runs):
            block = history[history["run_name"] == run].sort_values("epoch")
            color = SERIES[i % len(SERIES)]
            ax.plot(block["epoch"] + 1, block[column], color=color, label=labels.get(run, run))
        ax.set_title(panel_title)
        if "loss" in column:
            ax.set_yscale("log")
    for ax in axes[1]:
        ax.set_xlabel("Epoca")
    # Una sola leyenda para los cuatro paneles: la identidad de la serie no
    # depende del panel.
    handles, names = axes[0, 0].get_legend_handles_labels()
    ncol = min(4, max(1, len(names)))
    fig.legend(handles, names, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.04))
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return _finish(fig, path)


def plot_convergence(history: pd.DataFrame, labels: dict | None = None,
                     metric="val_top1", title="", ylabel="Top-1 en validacion", path=None):
    """Velocidad de convergencia: una curva por corrida, con etiqueta directa."""
    use_style()
    runs = list(dict.fromkeys(history["run_name"]))
    labels = labels or {r: r for r in runs}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, run in enumerate(runs):
        block = history[history["run_name"] == run].sort_values("epoch")
        color = SERIES[i % len(SERIES)]
        ax.plot(block["epoch"] + 1, block[metric], color=color)
        _label_last(ax, block["epoch"].iloc[-1] + 1, block[metric].iloc[-1],
                    labels.get(run, run), color)
    ax.set_xlabel("Epoca")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.margins(x=0.28)
    return _finish(fig, path)


# ---------------------------------------------------------------------------
# Transfer learning: el bloque principal
# ---------------------------------------------------------------------------


def plot_unfreeze_sweep(summary: pd.DataFrame, path=None):
    """Rendimiento frente a profundidad de descongelamiento.

    Tres metricas sobre el mismo eje porque las tres estan en la misma escala
    (fraccion de aciertos), con la barra de error de las condiciones que se
    corrieron con varias semillas.
    """
    use_style()
    data = summary[summary["pretrained"]].sort_values("k")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, (column, label) in enumerate([("top1", "Top-1"), ("top5", "Top-5"),
                                         ("macro_f1", "Macro-F1")]):
        color = SERIES[i]
        std = data.get(f"{column}_std")
        ax.errorbar(data["k"], data[column] * 100,
                    yerr=(std * 100 if std is not None else None),
                    color=color, marker="o", capsize=3, elinewidth=1)
        _label_last(ax, data["k"].iloc[-1], data[column].iloc[-1] * 100, label, color)

    scratch = summary[~summary["pretrained"]]
    if not scratch.empty:
        y = float(scratch["top1"].iloc[0]) * 100
        ax.axhline(y, color=INK_SOFT, linestyle="--", linewidth=1.2)
        ax.annotate("Desde cero (sin preentrenamiento)", xy=(0, y), xytext=(0, 5),
                    textcoords="offset points", fontsize=9, color=INK_SOFT)

    ax.set_xticks(range(5))
    ax.set_xticklabels(["0\ncongelado", "1", "2", "3", "4\nfull FT"])
    ax.set_xlabel("Etapas finales del backbone descongeladas (k)")
    ax.set_ylabel("Metrica en validacion (%)")
    ax.set_title("Rendimiento frente a profundidad de fine-tuning")
    ax.margins(x=0.22)
    return _finish(fig, path)


def plot_cost_frontier(summary: pd.DataFrame, x="gpu_hours",
                       xlabel="Costo de entrenamiento (horas de GPU)", path=None):
    """Frontera costo/rendimiento: Top-1 frente al costo, anotada por k.

    Es la figura que responde a si el fine-tuning parcial compra una mejor
    relacion entre rendimiento y costo: el punto que domina es el que esta
    arriba y a la izquierda.
    """
    use_style()
    data = summary[summary["pretrained"]].sort_values(x)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(data[x], data["top1"] * 100, color=SERIES[0], marker="o", zorder=2)
    for _, row in data.iterrows():
        label = "congelado" if row["k"] == 0 else ("full FT" if row["k"] == 4 else f"k={int(row['k'])}")
        ax.annotate(label, (row[x], row["top1"] * 100), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)

    scratch = summary[~summary["pretrained"]]
    if not scratch.empty:
        row = scratch.iloc[0]
        ax.scatter([row[x]], [row["top1"] * 100], color=SERIES[1], zorder=3, s=55)
        ax.annotate("desde cero", (row[x], row["top1"] * 100), xytext=(0, -16),
                    textcoords="offset points", ha="center", fontsize=9, color=SERIES[1])

    if x == "params_trainable_m":
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Top-1 en validacion (%)")
    ax.set_title("Frontera entre costo y rendimiento")
    ax.margins(x=0.18, y=0.22)
    return _finish(fig, path)


# ---------------------------------------------------------------------------
# Comparaciones categoricas
# ---------------------------------------------------------------------------


def plot_metric_bars(frame: pd.DataFrame, label_col: str, value_col: str,
                     title="", xlabel="", path=None, annotate_fmt="{:.1f}",
                     color_idx=0, baseline: float | None = None):
    """Barras horizontales ordenadas, con el valor escrito en cada barra."""
    use_style()
    data = frame.sort_values(value_col)
    fig, ax = plt.subplots(figsize=(7.5, max(2.6, 0.46 * len(data))))
    ax.barh(data[label_col], data[value_col], color=SERIES[color_idx], height=0.64)
    for label, value in zip(data[label_col], data[value_col]):
        ax.annotate(annotate_fmt.format(value), (value, label), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK_SOFT)
    if baseline is not None:
        ax.axvline(baseline, color=INK_SOFT, linestyle="--", linewidth=1.1)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.16)
    return _finish(fig, path)


def plot_efficiency_scatter(frame: pd.DataFrame, x: str, y: str, label_col: str,
                            xlabel: str, ylabel: str, title="", logx=False, path=None):
    """Nube de puntos con etiquetas directas; maximo tres categorias de color."""
    use_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(frame[x], frame[y], color=SERIES_SCATTER[0], s=70, zorder=3,
               edgecolor=SURFACE, linewidth=1.5)
    for _, row in frame.iterrows():
        ax.annotate(str(row[label_col]), (row[x], row[y]), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.margins(x=0.18, y=0.22)
    return _finish(fig, path)


# ---------------------------------------------------------------------------
# Long-tail
# ---------------------------------------------------------------------------


def plot_class_distribution(counts: np.ndarray, groups: np.ndarray | None = None,
                            title="Distribucion de imagenes por especie", path=None):
    """Curva de frecuencia por clase, ordenada; hace visible la forma long-tail."""
    use_style()
    order = np.argsort(-counts)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    if groups is None:
        ax.fill_between(range(len(counts)), counts[order], color=SERIES[0], alpha=0.85)
    else:
        ordered_groups = np.asarray(groups)[order]
        for i, group in enumerate(("head", "mid", "tail")):
            mask = ordered_groups == group
            idx = np.flatnonzero(mask)
            if idx.size:
                ax.fill_between(idx, counts[order][mask], color=SERIES[i], alpha=0.9)
                ax.annotate({"head": "frecuentes", "mid": "intermedias",
                             "tail": "minoritarias"}[group],
                            (idx.mean(), counts[order][mask].max()), xytext=(0, 6),
                            textcoords="offset points", ha="center", fontsize=9,
                            color=SERIES[i], fontweight="bold")
    ax.set_xlabel("Especies ordenadas por frecuencia")
    ax.set_ylabel("Imagenes de entrenamiento")
    ax.set_title(title)
    ax.margins(y=0.2)
    return _finish(fig, path)


def plot_group_recall(summary: pd.DataFrame, path=None):
    """Recall por grupo de frecuencia y estrategia: barras agrupadas.

    Es la lectura central del bloque long-tail: una mitigacion solo sirve si
    levanta la barra de las clases minoritarias sin hundir la de las frecuentes.
    """
    use_style()
    groups = ["head", "mid", "tail"]
    names = {"head": "Frecuentes", "mid": "Intermedias", "tail": "Minoritarias"}
    data = summary.dropna(subset=["tail_recall"])
    x = np.arange(len(data))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(7.5, 1.5 * len(data)), 4.6))
    for i, group in enumerate(groups):
        values = data[f"{group}_recall"] * 100
        bars = ax.bar(x + (i - 1) * width, values, width * 0.9,
                      color=SERIES[i], label=names[group])
        ax.bar_label(bars, fmt="%.1f", fontsize=8, color=INK_SOFT, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(data["name"], rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Recall medio por clase (%)")
    ax.set_title("Rendimiento por grupo de frecuencia")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.18)
    fig.tight_layout()
    return _finish(fig, path)


# ---------------------------------------------------------------------------
# Analisis de errores
# ---------------------------------------------------------------------------


def plot_confidence_histogram(correct_conf: np.ndarray, wrong_conf: np.ndarray, path=None):
    """Confianza del modelo separada por acierto y error.

    Los errores de alta confianza son los interesantes: indican que el modelo
    tiene una representacion equivocada, no simplemente incertidumbre.
    """
    use_style()
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(correct_conf, bins=bins, color=SERIES[0], alpha=0.85, label="Aciertos")
    ax.hist(wrong_conf, bins=bins, color=SERIES[1], alpha=0.75, label="Errores")
    ax.set_xlabel("Confianza de la prediccion (softmax maximo)")
    ax.set_ylabel("Imagenes")
    ax.set_title("Distribucion de confianza")
    ax.legend()
    return _finish(fig, path)


def plot_confusion_block(matrix: np.ndarray, labels: list[str],
                         title="Especies mas confundidas", path=None):
    """Submatriz de confusion de las clases con mas errores.

    El enunciado descarta la matriz completa de 10,000 x 10,000; esta vista
    muestra solo el bloque donde se concentran los errores, que es lo
    interpretable.
    """
    use_style()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", PALETTE_SEQ)
    fig, ax = plt.subplots(figsize=(max(6, 0.52 * len(labels)),
                                    max(5, 0.52 * len(labels))))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Clase real")
    ax.set_title(title)
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Imagenes")
    fig.tight_layout()
    return _finish(fig, path)
