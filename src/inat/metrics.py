"""Metricas para clasificacion multiclase de gran escala.

Con 10,000 especies el Accuracy agregado oculta casi todo lo interesante: una
clase mal aprendida mueve el top-1 en 0.01 puntos. Por eso se reportan ademas
Top-5, Macro-F1 (que pesa igual a cada especie) y Weighted-F1, y en el
escenario long-tail las mismas metricas desagregadas por grupo de frecuencia.

Los acumuladores trabajan sobre conteos por clase en GPU en lugar de guardar
todas las predicciones, de modo que evaluar 100,000 imagenes no obliga a
materializar una matriz de confusion de 10,000 x 10,000.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class ClassificationMeter:
    """Acumula estadisticas suficientes para top-k y F1 por clase.

    Para cada clase se guardan verdaderos positivos, total de predicciones
    (soporte predicho) y total real. Con esos tres vectores se derivan
    precision, recall y F1 por clase sin almacenar la matriz completa.
    """

    num_classes: int
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    topk: tuple = (1, 5)

    def __post_init__(self):
        z = lambda: torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
        self.tp = z()
        self.pred_count = z()
        self.true_count = z()
        self.correct_k = {k: 0 for k in self.topk}
        self.total = 0
        self.loss_sum = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float | None = None) -> None:
        batch = targets.numel()
        self.total += batch
        if loss is not None:
            self.loss_sum += float(loss) * batch

        maxk = max(self.topk)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        hits = pred.eq(targets.view(-1, 1))
        for k in self.topk:
            self.correct_k[k] += int(hits[:, :k].any(dim=1).sum())

        top1 = pred[:, 0]
        self.pred_count += torch.bincount(top1, minlength=self.num_classes)
        self.true_count += torch.bincount(targets, minlength=self.num_classes)
        correct_mask = top1.eq(targets)
        if correct_mask.any():
            self.tp += torch.bincount(targets[correct_mask], minlength=self.num_classes)

    # -- derivadas --------------------------------------------------------

    def per_class_f1(self) -> np.ndarray:
        """F1 por clase; 0 para clases sin soporte real ni predicciones."""
        tp = self.tp.double()
        precision = tp / self.pred_count.double().clamp(min=1)
        recall = tp / self.true_count.double().clamp(min=1)
        denom = (precision + recall).clamp(min=1e-12)
        f1 = 2 * precision * recall / denom
        return f1.cpu().numpy()

    def per_class_recall(self) -> np.ndarray:
        return (self.tp.double() / self.true_count.double().clamp(min=1)).cpu().numpy()

    def per_class_precision(self) -> np.ndarray:
        return (self.tp.double() / self.pred_count.double().clamp(min=1)).cpu().numpy()

    def compute(self) -> dict:
        """Metricas agregadas. El Macro-F1 solo promedia clases presentes en val."""
        present = (self.true_count > 0).cpu().numpy()
        f1 = self.per_class_f1()
        support = self.true_count.cpu().numpy()
        weights = support / max(support.sum(), 1)
        out = {
            "loss": self.loss_sum / max(self.total, 1),
            "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
            "weighted_f1": float((f1 * weights).sum()),
            "n": self.total,
        }
        for k in self.topk:
            out[f"top{k}"] = self.correct_k[k] / max(self.total, 1)
        return out

    def by_group(self, group_of_class: np.ndarray) -> dict:
        """Metricas separadas por grupo de frecuencia (head / mid / tail).

        El accuracy por grupo es el recall promedio de sus clases: es la
        lectura correcta cuando los grupos tienen soporte distinto en val.
        """
        recall = self.per_class_recall()
        f1 = self.per_class_f1()
        precision = self.per_class_precision()
        support = self.true_count.cpu().numpy()
        out: dict[str, dict] = {}
        for group in ("head", "mid", "tail"):
            mask = (group_of_class == group) & (support > 0)
            if not mask.any():
                continue
            out[group] = {
                "n_classes": int(mask.sum()),
                "support": int(support[mask].sum()),
                "recall": float(recall[mask].mean()),
                "precision": float(precision[mask].mean()),
                "macro_f1": float(f1[mask].mean()),
            }
        return out

    def by_taxon(self, taxon_of_class: np.ndarray) -> dict:
        """Macro-F1 y recall por grupo taxonomico (Aves, Insecta, Plantae...)."""
        recall = self.per_class_recall()
        f1 = self.per_class_f1()
        support = self.true_count.cpu().numpy()
        out: dict[str, dict] = {}
        for taxon in np.unique(taxon_of_class):
            mask = (taxon_of_class == taxon) & (support > 0)
            if not mask.any():
                continue
            out[str(taxon)] = {
                "n_classes": int(mask.sum()),
                "recall": float(recall[mask].mean()),
                "macro_f1": float(f1[mask].mean()),
            }
        return out
