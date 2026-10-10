"""Control centralizado de aleatoriedad.

Toda fuente de aleatoriedad del proyecto (muestreo de clases long-tail,
inicialización de pesos, orden del DataLoader, augmentation) se deriva de una
única semilla declarada en la configuración del experimento, de modo que una
corrida pueda repetirse exactamente con el mismo `ExperimentConfig`.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Fija las semillas de Python, NumPy y PyTorch.

    `deterministic=True` fuerza algoritmos deterministas de cuDNN. Tiene un
    costo de velocidad apreciable en convoluciones, por lo que el proyecto lo
    usa solo en las pruebas de integridad y no en el entrenamiento masivo: para
    los experimentos se reporta la media sobre varias semillas en lugar de
    exigir reproducibilidad bit a bit.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """Semilla distinta y reproducible por worker del DataLoader."""
    base = torch.initial_seed() % (2 ** 31)
    seed = (base + worker_id) % (2 ** 31)
    random.seed(seed)
    np.random.seed(seed)


def make_generator(seed: int) -> torch.Generator:
    """Generador explícito para el shuffling del DataLoader."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
