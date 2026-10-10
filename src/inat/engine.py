"""Bucle de entrenamiento, evaluacion y contabilidad de costo computacional.

Resuelve tres cosas que el proyecto necesita de forma transversal:

1. Entrenar y validar con precision mixta sobre A100, midiendo tiempo y
   memoria para que la comparacion rendimiento/costo sea cuantitativa.
2. Early stopping sobre validacion con recuperacion del mejor modelo.
3. Sobrevivir al limite de 8 horas por job de Slurm: el estado completo se
   guarda por epoca y la corrida se reanuda donde quedo.
"""
from __future__ import annotations

import json
import math
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ExperimentConfig
from .metrics import ClassificationMeter
from .models import count_parameters, set_frozen_bn_eval

# Slurm envia SIGUSR1 antes de matar el job (--signal=USR1@...). Al recibirla
# se marca la bandera y el bucle cierra ordenadamente tras la epoca en curso.
_STOP_REQUESTED = False


def _handle_signal(signum, frame):  # pragma: no cover - depende de Slurm
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def install_signal_handler() -> None:
    # SIGUSR1 no existe en Windows; el notebook debe poder importarse igual en
    # el entorno local de desarrollo que en el nodo de computo.
    for name in ("SIGUSR1", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError):
            pass  # no instalable desde un hilo secundario


def stop_requested() -> bool:
    return _STOP_REQUESTED


# ---------------------------------------------------------------------------
# Perdidas
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Focal loss multiclase.

    Reescala la cross-entropy por `(1-p_t)^gamma`, reduciendo el peso de los
    ejemplos ya bien clasificados. En un escenario long-tail eso desplaza la
    capacidad del modelo hacia las especies dificiles, que son en su mayoria
    las minoritarias.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, target, weight=self.weight,
            label_smoothing=self.label_smoothing, reduction="none",
        )
        pt = torch.exp(-ce).clamp(max=1.0)
        return ((1 - pt) ** self.gamma * ce).mean()


def class_weights(counts: np.ndarray, mode: str, beta: float = 0.999) -> torch.Tensor | None:
    """Pesos por clase para la cross-entropy ponderada.

    - `inverse`: proporcional a 1/n_c. Directo, pero sobre-corrige cuando
      algunas clases tienen muy pocos ejemplos.
    - `effective`: numero efectivo de muestras de Cui et al. (2019),
      `(1-beta)/(1-beta^n_c)`. Interpola entre no ponderar y 1/n_c y es mas
      estable cuando el desbalance es extremo.

    En ambos casos los pesos se normalizan a media 1 para no alterar la escala
    efectiva del learning rate entre experimentos.
    """
    if mode == "none":
        return None
    counts = np.maximum(counts, 1).astype(np.float64)
    if mode == "inverse":
        w = 1.0 / counts
    elif mode == "effective":
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    else:
        raise ValueError(f"class_weighting desconocido: {mode}")
    w = w / w.mean()
    return torch.as_tensor(w, dtype=torch.float32)


def build_criterion(cfg: ExperimentConfig, counts: np.ndarray, device: torch.device) -> nn.Module:
    weight = class_weights(counts, cfg.optim.class_weighting)
    if weight is not None:
        weight = weight.to(device)
    if cfg.optim.loss == "focal":
        return FocalLoss(cfg.optim.focal_gamma, weight, cfg.optim.label_smoothing).to(device)
    # "ce" y "weighted_ce" comparten implementacion: la diferencia es el peso.
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.optim.label_smoothing).to(device)


# ---------------------------------------------------------------------------
# Optimizador y schedule
# ---------------------------------------------------------------------------


def build_optimizer(model: nn.Module, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    """Crea el optimizador solo sobre los parametros entrenables.

    Cuando el backbone esta parcial o totalmente congelado, pasarle los
    parametros congelados al optimizador reservaria estado (momentum, momentos
    de Adam) para tensores que nunca se actualizan: inflaria la memoria y
    distorsionaria la medicion de costo que el proyecto quiere comparar.

    `backbone_lr_mult` permite un learning rate menor en el backbone que en la
    cabeza, la practica habitual en fine-tuning para no destruir las
    representaciones preentrenadas en los primeros pasos.
    """
    o = cfg.optim
    head_prefixes = ("fc.", "classifier.")
    backbone, head = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (head if name.startswith(head_prefixes) else backbone).append(param)

    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": o.lr * o.backbone_lr_mult})
    if head:
        groups.append({"params": head, "lr": o.lr})
    if not groups:
        raise ValueError("El modelo no tiene parametros entrenables.")

    wd = o.weight_decay
    name = o.optimizer
    if name == "sgd":
        return torch.optim.SGD(groups, lr=o.lr, momentum=0.0, weight_decay=wd)
    if name == "sgd_momentum":
        return torch.optim.SGD(groups, lr=o.lr, momentum=o.momentum, weight_decay=wd,
                               nesterov=o.nesterov)
    if name == "adam":
        return torch.optim.Adam(groups, lr=o.lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=o.lr, weight_decay=wd)
    if name == "rmsprop":
        return torch.optim.RMSprop(groups, lr=o.lr, momentum=o.momentum, weight_decay=wd)
    if name == "adagrad":
        return torch.optim.Adagrad(groups, lr=o.lr, weight_decay=wd)
    raise ValueError(f"Optimizador desconocido: {name}")


def build_scheduler(optimizer, cfg: ExperimentConfig, steps_per_epoch: int):
    """Schedule por paso con warmup lineal.

    El warmup importa sobre todo en los experimentos de fine-tuning: la cabeza
    arranca con pesos aleatorios y sus gradientes iniciales son grandes; sin
    warmup esos primeros pasos se propagan al backbone preentrenado y borran
    parte de lo aprendido en ImageNet.
    """
    total = max(cfg.optim.epochs * steps_per_epoch, 1)
    warmup = max(cfg.optim.warmup_epochs * steps_per_epoch, 0)

    def lr_lambda(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if cfg.optim.scheduler == "none":
            return 1.0
        progress = (step - warmup) / max(total - warmup, 1)
        if cfg.optim.scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        if cfg.optim.scheduler == "step":
            return 0.1 ** int(min(progress, 0.999) * 3)
        raise ValueError(cfg.optim.scheduler)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Estado y resultado de una corrida
# ---------------------------------------------------------------------------


@dataclass
class EpochRecord:
    """Una fila de la curva de aprendizaje."""

    epoch: int
    lr: float
    train_loss: float
    train_top1: float
    val_loss: float
    val_top1: float
    val_top5: float
    val_macro_f1: float
    val_weighted_f1: float
    train_seconds: float
    val_seconds: float
    peak_memory_gb: float


@dataclass
class RunResult:
    """Todo lo que una corrida produce y el informe necesita."""

    exp_id: str
    name: str
    group: str
    seed: int
    config: dict
    history: list = field(default_factory=list)
    best_epoch: int = -1
    best: dict = field(default_factory=dict)
    final: dict = field(default_factory=dict)
    by_group: dict = field(default_factory=dict)
    by_taxon: dict = field(default_factory=dict)
    params_total: int = 0
    params_trainable: int = 0
    train_seconds: float = 0.0
    gpu_hours: float = 0.0
    epochs_run: int = 0
    stopped_early: bool = False
    peak_memory_gb: float = 0.0
    throughput_img_s: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# ---------------------------------------------------------------------------
# Bucles
# ---------------------------------------------------------------------------


def _wants_channels_last(cfg: ExperimentConfig) -> bool:
    """channels_last solo beneficia a las convoluciones.

    En el MLP la imagen se aplana de inmediato, asi que reordenar la memoria es
    trabajo puro sin ganancia.
    """
    return cfg.optim.channels_last and cfg.model.arch != "mlp"


def _autocast(device: torch.device, enabled: bool):
    """bfloat16 en A100: mismo rango dinamico que fp32, sin necesidad de GradScaler."""
    dtype = torch.bfloat16 if (enabled and device.type == "cuda") else torch.float32
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled and device.type == "cuda")


def train_one_epoch(model, loader, criterion, optimizer, scheduler, cfg, device, num_classes):
    model.train()
    set_frozen_bn_eval(model, cfg.model)
    meter = ClassificationMeter(num_classes, device, topk=(1,))
    channels_last = _wants_channels_last(cfg)
    start = time.perf_counter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.to(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, cfg.optim.amp):
            logits = model(images)
            loss = criterion(logits, targets)
        loss.backward()
        if cfg.optim.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.optim.grad_clip
            )
        optimizer.step()
        scheduler.step()
        meter.update(logits.detach().float(), targets, loss.item())

    if device.type == "cuda":
        torch.cuda.synchronize()
    return meter.compute(), time.perf_counter() - start


@torch.no_grad()
def evaluate(model, loader, criterion, cfg, device, num_classes, topk=(1, 5)):
    model.eval()
    meter = ClassificationMeter(num_classes, device, topk=topk)
    channels_last = _wants_channels_last(cfg)
    start = time.perf_counter()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.to(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        with _autocast(device, cfg.optim.amp):
            logits = model(images)
            loss = criterion(logits, targets)
        meter.update(logits.float(), targets, loss.item())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return meter, time.perf_counter() - start


class EarlyStopping:
    """Detiene el entrenamiento cuando la metrica de validacion deja de mejorar.

    Guarda ademas los pesos de la mejor epoca en CPU, para que el modelo que se
    evalua al final sea el de mejor validacion y no el de la ultima epoca.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.best = -math.inf if cfg.mode == "max" else math.inf
        self.best_epoch = -1
        self.bad_epochs = 0
        self.best_state: dict | None = None

    def _improved(self, value: float) -> bool:
        if self.cfg.mode == "max":
            return value > self.best + self.cfg.min_delta
        return value < self.best - self.cfg.min_delta

    def step(self, value: float, epoch: int, model: nn.Module) -> bool:
        """Devuelve True si hay que detener el entrenamiento."""
        if self._improved(value):
            self.best, self.best_epoch, self.bad_epochs = value, epoch, 0
            if self.cfg.restore_best:
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.bad_epochs += 1
        return self.cfg.enabled and self.bad_epochs >= self.cfg.patience

    def restore(self, model: nn.Module) -> None:
        if self.cfg.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: ExperimentConfig,
    device: torch.device,
    num_classes: int,
    class_counts: np.ndarray,
    group_of_class: np.ndarray | None = None,
    taxon_of_class: np.ndarray | None = None,
    checkpoint_path: Path | None = None,
    log_fn=print,
) -> RunResult:
    """Entrena un experimento completo y devuelve su `RunResult`.

    Si `checkpoint_path` existe, la corrida se reanuda desde ahi: es lo que
    permite que un experimento largo atraviese varios jobs de 8 horas.
    """
    if _wants_channels_last(cfg):
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    criterion = build_criterion(cfg, class_counts, device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    stopper = EarlyStopping(cfg.early_stop)
    params = count_parameters(model)

    result = RunResult(
        exp_id=cfg.exp_id, name=cfg.name, group=cfg.group, seed=cfg.seed,
        config=cfg.to_dict(), params_total=params.total, params_trainable=params.trainable,
    )
    start_epoch = 0

    state = None
    if checkpoint_path is not None and checkpoint_path.exists():
        try:
            state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as exc:
            # Un job cortado durante la escritura puede dejar el archivo
            # truncado. Perder un checkpoint cuesta una corrida; abortar la fase
            # por el deja en cola todas las demas.
            log_fn(f"  checkpoint ilegible ({exc}); se reinicia {cfg.run_name}")
            checkpoint_path.unlink(missing_ok=True)

    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        stopper.best = state["best"]
        stopper.best_epoch = state["best_epoch"]
        stopper.bad_epochs = state["bad_epochs"]
        stopper.best_state = state.get("best_state")
        result.history = state["history"]
        result.train_seconds = state["train_seconds"]
        start_epoch = state["epoch"] + 1
        log_fn(f"  reanudando {cfg.run_name} desde la epoca {start_epoch}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    n_train = len(train_loader.dataset)
    for epoch in range(start_epoch, cfg.optim.epochs):
        train_stats, train_time = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, cfg, device, num_classes
        )
        val_meter, val_time = evaluate(model, val_loader, criterion, cfg, device, num_classes)
        val_stats = val_meter.compute()
        peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0

        record = EpochRecord(
            epoch=epoch,
            lr=optimizer.param_groups[-1]["lr"],
            train_loss=train_stats["loss"], train_top1=train_stats["top1"],
            val_loss=val_stats["loss"], val_top1=val_stats["top1"], val_top5=val_stats["top5"],
            val_macro_f1=val_stats["macro_f1"], val_weighted_f1=val_stats["weighted_f1"],
            train_seconds=train_time, val_seconds=val_time, peak_memory_gb=peak,
        )
        result.history.append(asdict(record))
        result.train_seconds += train_time + val_time
        result.epochs_run = epoch + 1
        result.peak_memory_gb = max(result.peak_memory_gb, peak)

        log_fn(
            f"  ep {epoch + 1:>2}/{cfg.optim.epochs}  "
            f"train {train_stats['loss']:.3f}/{train_stats['top1']:.3f}  "
            f"val {val_stats['loss']:.3f}/{val_stats['top1']:.3f} top5 {val_stats['top5']:.3f} "
            f"mF1 {val_stats['macro_f1']:.3f}  {train_time:.0f}s"
        )

        monitored = val_stats[cfg.early_stop.metric.replace("val_", "")]
        should_stop = stopper.step(monitored, epoch, model)

        if checkpoint_path is not None:
            tmp = checkpoint_path.with_suffix(".tmp")
            torch.save(
                {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "best": stopper.best, "best_epoch": stopper.best_epoch,
                    "bad_epochs": stopper.bad_epochs, "best_state": stopper.best_state,
                    "history": result.history, "train_seconds": result.train_seconds,
                },
                tmp,
            )
            tmp.replace(checkpoint_path)

        if should_stop:
            result.stopped_early = True
            log_fn(f"  early stopping en la epoca {epoch + 1} (mejor: {stopper.best_epoch + 1})")
            break
        if stop_requested():
            log_fn("  senal de Slurm recibida; se guarda estado y se sale")
            raise KeyboardInterrupt("preempted")

    # Evaluacion final con los pesos de la mejor epoca.
    stopper.restore(model)
    final_meter, _ = evaluate(model, val_loader, criterion, cfg, device, num_classes)
    result.final = final_meter.compute()
    result.best_epoch = stopper.best_epoch
    result.best = {"val_" + k: v for k, v in result.final.items()}
    if group_of_class is not None:
        result.by_group = final_meter.by_group(group_of_class)
    if taxon_of_class is not None:
        result.by_taxon = final_meter.by_taxon(taxon_of_class)
    result.gpu_hours = result.train_seconds / 3600
    if result.train_seconds > 0:
        result.throughput_img_s = n_train * result.epochs_run / result.train_seconds
    return result
