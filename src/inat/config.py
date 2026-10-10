"""Configuración declarativa de un experimento.

Un `ExperimentConfig` contiene *todo* lo que distingue una corrida de otra:
arquitectura, resolución, optimizador, regularización, augmentation, estrategia
de transfer learning y escenario de datos. El runner serializa el objeto a JSON
junto a los resultados, de modo que cada fila de la tabla final sea trazable
hasta sus hiperparámetros exactos.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_data_root() -> Path:
    """Raíz del dataset preprocesado.

    En Khipu se resuelve por variable de entorno; en local cae al subset de
    desarrollo para que el notebook pueda abrirse sin el corpus completo.
    """
    env = os.environ.get("INAT_DATA_ROOT")
    if env:
        return Path(env)
    return Path("data/inat2021")


@dataclass(frozen=True)
class Paths:
    """Rutas del proyecto. Separa datos (grandes, fuera del repo) de salidas."""

    data_root: Path = field(default_factory=_default_data_root)
    results: Path = field(default_factory=lambda: Path(os.environ.get("INAT_RESULTS", "results")))
    figures: Path = field(default_factory=lambda: Path(os.environ.get("INAT_FIGURES", "figures")))

    @property
    def images(self) -> Path:
        """Imágenes ya redimensionadas (lado corto fijo) usadas en entrenamiento."""
        return self.data_root / "images"

    @property
    def raw(self) -> Path:
        """Descarga original sin procesar (solo se usa en la fase de preparación)."""
        return self.data_root / "raw"

    @property
    def index(self) -> Path:
        """Índices tabulares imagen->clase y metadatos taxonómicos."""
        return self.data_root / "index"


# --------------------------------------------------------------------------
# Hiperparámetros agrupados por bloque temático del enunciado
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DataConfig:
    """Qué datos ve el modelo y cómo se transforman."""

    image_size: int = 160              # resolución de entrada a la red
    num_classes: int | None = None     # None => todas las especies del escenario
    scenario: str = "balanced"         # "balanced" | "longtail"
    longtail_seed: int = 42            # semilla del muestreo long-tail
    longtail_head_frac: float = 0.20   # fracción de especies "frecuentes"
    longtail_mid_frac: float = 0.30    # fracción de especies "intermedias"
    longtail_head_n: int = 50          # imágenes/clase en el grupo frecuente
    longtail_mid_n: int = 20           # imágenes/clase en el grupo intermedio
    longtail_tail_n: int = 5           # imágenes/clase en el grupo minoritario
    batch_size: int = 256
    eval_batch_size: int = 512
    num_workers: int = 12
    prefetch_factor: int = 4
    persistent_workers: bool = True
    # Data augmentation (cada flag es una variable del ablation study)
    random_resized_crop: bool = True
    rrc_scale_min: float = 0.35
    horizontal_flip: bool = True
    color_jitter: float = 0.0          # intensidad; 0 desactiva
    rotation_degrees: float = 0.0
    random_erasing: float = 0.0        # probabilidad; 0 desactiva
    # Submuestreo del corpus para depuración; None = corpus completo
    subset_classes: int | None = None
    subset_images_per_class: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    """Arquitectura y cómo se inicializa."""

    arch: str = "resnet50"             # ver inat.models.build_model
    pretrained: bool = False           # pesos ImageNet
    dropout: float = 0.0               # dropout en la cabeza de clasificación
    batch_norm: bool = True            # solo aplica al MLP; las CNN lo traen fijo
    mlp_hidden: tuple[int, ...] = (2048, 1024)
    # Transfer learning: qué parte del backbone se entrena
    freeze_mode: str = "none"          # "none" | "full" | "partial"
    partial_unfrozen_blocks: int = 1   # nº de bloques finales descongelados
    freeze_bn_stats: bool = True       # congelar estadísticas BN del backbone congelado


@dataclass(frozen=True)
class OptimConfig:
    """Optimización, schedule y regularización explícita."""

    optimizer: str = "sgd"             # "sgd" | "sgd_momentum" | "adam" | "adamw" | "rmsprop" | "adagrad"
    lr: float = 0.1
    backbone_lr_mult: float = 1.0      # lr diferencial backbone vs cabeza
    momentum: float = 0.9
    weight_decay: float = 0.0
    nesterov: bool = False
    scheduler: str = "cosine"          # "cosine" | "step" | "none"
    warmup_epochs: int = 1
    epochs: int = 15
    label_smoothing: float = 0.0
    grad_clip: float = 0.0
    amp: bool = True                   # precisión mixta (bf16 en A100)
    channels_last: bool = True
    # Estrategia frente al desbalance
    loss: str = "ce"                   # "ce" | "weighted_ce" | "focal"
    focal_gamma: float = 2.0
    class_weighting: str = "none"      # "none" | "inverse" | "effective"
    weighted_sampler: bool = False


@dataclass(frozen=True)
class EarlyStopConfig:
    """Criterio de early stopping sobre validación."""

    enabled: bool = True
    metric: str = "val_top1"           # métrica monitoreada
    mode: str = "max"
    patience: int = 4                  # épocas sin mejora antes de detener
    min_delta: float = 1e-4            # mejora mínima para resetear la paciencia
    restore_best: bool = True          # recuperar los pesos de la mejor época


@dataclass(frozen=True)
class ExperimentConfig:
    """Un experimento completo y reproducible."""

    exp_id: str
    name: str
    group: str                         # bloque del informe al que pertenece
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    early_stop: EarlyStopConfig = field(default_factory=EarlyStopConfig)
    notes: str = ""
    #: Conservar los pesos de la mejor epoca al terminar. Se activa solo en
    #: los experimentos que alimentan el analisis de errores: guardar las 54
    #: corridas ocuparia varios gigabytes sin aportar nada al informe.
    keep_weights: bool = False

    # -- serialización ----------------------------------------------------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)

    @property
    def fingerprint(self) -> str:
        """Hash corto de la configuración; detecta corridas incomparables."""
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @property
    def run_name(self) -> str:
        return f"{self.exp_id}_{self.name}_s{self.seed}"

    def replace(self, **kwargs) -> "ExperimentConfig":
        """Copia con campos modificados; acepta `data__image_size=224`."""
        nested: dict[str, dict] = {}
        top: dict[str, object] = {}
        for key, value in kwargs.items():
            if "__" in key:
                block, attr = key.split("__", 1)
                nested.setdefault(block, {})[attr] = value
            else:
                top[key] = value
        for block, attrs in nested.items():
            top[block] = dataclasses.replace(getattr(self, block), **attrs)
        return dataclasses.replace(self, **top)
