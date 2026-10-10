"""Analisis de errores del mejor modelo.

Vuelve a evaluar el modelo guardando las predicciones, y a partir de ellas
responde las preguntas interpretativas del enunciado: si las especies
confundidas comparten genero, que grupos taxonomicos son mas dificiles, como se
reparte la confianza entre aciertos y errores, y si las clases minoritarias
fallan de una forma distinta.

No se construye la matriz de confusion completa de 10,000 x 10,000: se
acumulan solo los pares (real, predicha) que aparecen, que son ordenes de
magnitud menos.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig, Paths
from .data import Scenario, build_loaders


@dataclass
class Predictions:
    """Predicciones sobre validacion, suficientes para todo el analisis."""

    targets: np.ndarray        # clase real
    top5: np.ndarray           # (N, 5) clases mas probables
    confidence: np.ndarray     # softmax de la clase predicha
    true_prob: np.ndarray      # probabilidad asignada a la clase real
    file_names: np.ndarray

    @property
    def predicted(self) -> np.ndarray:
        return self.top5[:, 0]

    @property
    def correct(self) -> np.ndarray:
        return self.predicted == self.targets


@torch.no_grad()
def collect_predictions(model, scenario: Scenario, cfg: ExperimentConfig,
                        paths: Paths, device: torch.device) -> Predictions:
    """Evalua el modelo sobre validacion conservando las predicciones."""
    _, val_loader = build_loaders(scenario, cfg.data, paths, cfg.seed)
    model = model.to(device).eval()

    targets, top5, confidence, true_prob = [], [], [], []
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        if cfg.optim.channels_last:
            images = images.to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=cfg.optim.amp and device.type == "cuda"):
            logits = model(images)
        probs = torch.softmax(logits.float(), dim=1)
        values, idx = probs.topk(5, dim=1)
        targets.append(labels.cpu().numpy())
        top5.append(idx.cpu().numpy())
        confidence.append(values[:, 0].cpu().numpy())
        true_prob.append(probs.gather(1, labels.view(-1, 1)).squeeze(1).cpu().numpy())

    return Predictions(
        targets=np.concatenate(targets), top5=np.concatenate(top5),
        confidence=np.concatenate(confidence), true_prob=np.concatenate(true_prob),
        file_names=scenario.val["file_name"].to_numpy(),
    )


# ---------------------------------------------------------------------------
# Patrones de error
# ---------------------------------------------------------------------------


def confusion_pairs(pred: Predictions, categories: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    """Pares de especies mas confundidos, con su relacion taxonomica.

    La columna `mismo_genero` responde directamente a la pregunta del enunciado
    sobre si el modelo confunde especies emparentadas.
    """
    info = categories.set_index("label")
    wrong = ~pred.correct
    counts = Counter(zip(pred.targets[wrong], pred.predicted[wrong]))

    rows = []
    for (real, predicted), n in counts.most_common(top):
        a, b = info.loc[real], info.loc[predicted]
        rows.append({
            "especie_real": a["name"], "especie_predicha": b["name"],
            "errores": n,
            "mismo_genero": bool(a.get("genus") == b.get("genus")),
            "misma_familia": bool(a.get("family") == b.get("family")),
            "mismo_orden": bool(a.get("order") == b.get("order")),
            "misma_clase": bool(a.get("class") == b.get("class")),
            "label_real": int(real), "label_predicha": int(predicted),
        })
    return pd.DataFrame(rows)


def taxonomic_error_profile(pred: Predictions, categories: pd.DataFrame) -> pd.DataFrame:
    """Que proporcion de los errores se queda dentro de cada nivel taxonomico.

    Si la mayor parte de los errores cae dentro del mismo genero o familia, el
    modelo aprendio la estructura gruesa del arbol y falla solo en la
    discriminacion fina; si los errores cruzan clases enteras, el fallo es de
    representacion basica.
    """
    info = categories.set_index("label")
    wrong = ~pred.correct
    real = pred.targets[wrong]
    predicted = pred.predicted[wrong]
    total = int(wrong.sum())
    if total == 0:
        return pd.DataFrame()

    rows = []
    for rank in ("genus", "family", "order", "class", "phylum"):
        if rank not in info.columns:
            continue
        same = (info.loc[real, rank].to_numpy() == info.loc[predicted, rank].to_numpy()).sum()
        rows.append({"nivel": rank, "errores_dentro_del_nivel": int(same),
                     "proporcion": same / total})
    return pd.DataFrame(rows)


def confidence_analysis(pred: Predictions, high: float = 0.8, low: float = 0.4) -> pd.DataFrame:
    """Reparte las predicciones en las cuatro categorias que pide el enunciado."""
    correct = pred.correct
    conf = pred.confidence
    buckets = {
        "Correctas con alta confianza": correct & (conf >= high),
        "Correctas con baja confianza": correct & (conf < low),
        "Incorrectas con alta confianza": ~correct & (conf >= high),
        "Incorrectas con baja confianza": ~correct & (conf < low),
    }
    rows = [{"categoria": name, "n": int(mask.sum()),
             "porcentaje": 100 * mask.sum() / len(conf),
             "confianza_media": float(conf[mask].mean()) if mask.any() else np.nan}
            for name, mask in buckets.items()]
    return pd.DataFrame(rows)


def per_taxon_accuracy(pred: Predictions, categories: pd.DataFrame,
                       rank: str = "class") -> pd.DataFrame:
    """Exactitud por grupo taxonomico; identifica los grupos mas dificiles."""
    info = categories.set_index("label")
    taxon = info.loc[pred.targets, rank].to_numpy()
    frame = pd.DataFrame({"taxon": taxon, "correct": pred.correct,
                          "top5": (pred.top5 == pred.targets[:, None]).any(axis=1),
                          "confianza": pred.confidence})
    out = frame.groupby("taxon").agg(
        n=("correct", "size"), top1=("correct", "mean"),
        top5=("top5", "mean"), confianza=("confianza", "mean"),
    ).sort_values("top1")
    return out.reset_index()


def group_error_profile(pred: Predictions, group_of_class: np.ndarray) -> pd.DataFrame:
    """Compara el tipo de error entre clases frecuentes y minoritarias.

    `prediccion_hacia_head` mide el sesgo caracteristico del desbalance: los
    errores de las clases minoritarias tienden a caer en clases frecuentes
    porque el clasificador aprendio un prior que las favorece.
    """
    rows = []
    for group in ("head", "mid", "tail"):
        mask = np.isin(pred.targets, np.flatnonzero(group_of_class == group))
        if not mask.any():
            continue
        wrong = mask & ~pred.correct
        toward_head = (
            np.isin(pred.predicted[wrong], np.flatnonzero(group_of_class == "head")).mean()
            if wrong.any() else np.nan
        )
        rows.append({
            "grupo": group, "n_imagenes": int(mask.sum()),
            "top1": float(pred.correct[mask].mean()),
            "top5": float((pred.top5[mask] == pred.targets[mask, None]).any(axis=1).mean()),
            "confianza_media": float(pred.confidence[mask].mean()),
            "prediccion_hacia_head": float(toward_head),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ejemplos para inspeccion visual
# ---------------------------------------------------------------------------


def example_images(pred: Predictions, categories: pd.DataFrame, paths: Paths,
                   kind: str, n: int = 8, size: int = 200, seed: int = 42):
    """Selecciona ejemplos representativos de un tipo de prediccion.

    `kind` admite: 'correcta_alta', 'correcta_baja', 'incorrecta_alta',
    'atipica' (la clase real recibio una probabilidad minima).
    """
    from PIL import Image

    correct, conf = pred.correct, pred.confidence
    if kind == "correcta_alta":
        pool = np.flatnonzero(correct & (conf >= 0.8))
        order = pool[np.argsort(-conf[pool])]
    elif kind == "correcta_baja":
        pool = np.flatnonzero(correct)
        order = pool[np.argsort(conf[pool])]
    elif kind == "incorrecta_alta":
        pool = np.flatnonzero(~correct)
        order = pool[np.argsort(-conf[pool])]
    elif kind == "atipica":
        order = np.argsort(pred.true_prob)
    else:
        raise ValueError(f"tipo desconocido: {kind}")

    info = categories.set_index("label")
    images, captions = [], []
    for idx in order[:n]:
        with Image.open(paths.images / pred.file_names[idx]) as img:
            images.append(img.convert("RGB").resize((size, size), Image.BICUBIC))
        real = info.loc[pred.targets[idx], "name"]
        predicted = info.loc[pred.predicted[idx], "name"]
        if correct[idx]:
            captions.append(f"{real}\np={conf[idx]:.2f}")
        else:
            captions.append(f"real: {real}\npred: {predicted} (p={conf[idx]:.2f})")
    return images, captions


def confusion_block(pred: Predictions, categories: pd.DataFrame, n_classes: int = 15):
    """Submatriz de confusion de las clases con mas errores."""
    wrong = ~pred.correct
    worst = pd.Series(pred.targets[wrong]).value_counts().head(n_classes).index.to_numpy()
    index = {label: i for i, label in enumerate(worst)}
    matrix = np.zeros((len(worst), len(worst)), dtype=int)
    mask = np.isin(pred.targets, worst) & np.isin(pred.predicted, worst)
    for real, predicted in zip(pred.targets[mask], pred.predicted[mask]):
        matrix[index[real], index[predicted]] += 1
    info = categories.set_index("label")
    labels = [info.loc[label, "name"] for label in worst]
    return matrix, labels
