"""Arquitecturas y estrategias de transfer learning.

Contiene el baseline MLP, los constructores de las CNN modernas de torchvision
adaptadas a C especies, y las tres estrategias de congelamiento que sostienen
la pregunta de investigacion del proyecto (backbone congelado, fine-tuning
parcial y fine-tuning completo).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.models as tvm

from .config import ModelConfig

# Arquitecturas soportadas -> (constructor, enum de pesos ImageNet).
ARCHS = {
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2),
    "densenet121": (tvm.densenet121, tvm.DenseNet121_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "convnext_tiny": (tvm.convnext_tiny, tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": (tvm.mobilenet_v3_large, tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
}


# ---------------------------------------------------------------------------
# Baseline completamente conectado
# ---------------------------------------------------------------------------


class MLPBaseline(nn.Module):
    """Perceptron multicapa sobre la imagen aplanada.

    Existe para cuantificar el costo de descartar la estructura espacial: al
    vectorizar la imagen se pierde la localidad de los pixeles y la invarianza
    a traslacion, y el numero de parametros de la primera capa crece de forma
    cuadratica con el lado de la entrada. Ambos efectos se miden en la Parte 2.
    """

    def __init__(
        self,
        image_size: int,
        num_classes: int,
        hidden=(2048, 1024),
        dropout: float = 0.0,
        batch_norm: bool = True,
    ):
        super().__init__()
        in_features = 3 * image_size * image_size
        layers: list[nn.Module] = [nn.Flatten()]
        prev = in_features
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _init_weights(module: nn.Module) -> None:
    """Inicializacion He para capas lineales, apropiada para activaciones ReLU."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# CNN modernas
# ---------------------------------------------------------------------------


def _replace_classifier(model: nn.Module, arch: str, num_classes: int, dropout: float) -> nn.Module:
    """Sustituye la cabeza de 1000 clases de ImageNet por una de C especies."""

    def head(in_features: int) -> nn.Module:
        if dropout > 0:
            return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return nn.Linear(in_features, num_classes)

    if arch.startswith("resnet"):
        model.fc = head(model.fc.in_features)
    elif arch.startswith("densenet"):
        model.classifier = head(model.classifier.in_features)
    elif arch.startswith("efficientnet"):
        model.classifier[-1] = head(model.classifier[-1].in_features)
    elif arch.startswith("convnext"):
        model.classifier[-1] = head(model.classifier[-1].in_features)
    elif arch.startswith("mobilenet"):
        model.classifier[-1] = head(model.classifier[-1].in_features)
    else:
        raise ValueError(f"No se sabe reemplazar la cabeza de {arch}")
    return model


def classifier_parameter_names(model: nn.Module, arch: str) -> list[str]:
    """Nombres de los parametros que forman la cabeza de clasificacion."""
    prefix = {"resnet": "fc", "densenet": "classifier", "efficientnet": "classifier",
              "convnext": "classifier", "mobilenet": "classifier"}
    for key, value in prefix.items():
        if arch.startswith(key):
            return [n for n, _ in model.named_parameters() if n.startswith(value)]
    raise ValueError(arch)


def _backbone_stages(model: nn.Module, arch: str) -> list[nn.Module]:
    """Etapas del backbone ordenadas de la mas temprana a la mas tardia.

    El fine-tuning parcial descongela las ultimas `k` de esta lista. Se trabaja
    con etapas y no con capas sueltas porque dentro de un bloque residual o
    densamente conectado las capas comparten una misma escala de
    representacion; partirlo por la mitad mezclaria niveles de abstraccion.
    """
    if arch.startswith("resnet"):
        return [model.layer1, model.layer2, model.layer3, model.layer4]
    if arch.startswith("densenet"):
        feats = model.features
        return [feats.denseblock1, feats.denseblock2, feats.denseblock3, feats.denseblock4]
    if arch.startswith(("efficientnet", "convnext", "mobilenet")):
        # `features` es un Sequential de bloques; se agrupan en cuatro etapas
        # de tamano comparable para que `partial_unfrozen_blocks` sea
        # interpretable de la misma forma en todas las arquitecturas.
        blocks = list(model.features.children())
        step = max(1, len(blocks) // 4)
        return [nn.ModuleList(blocks[i : i + step]) for i in range(0, len(blocks), step)][-4:]
    raise ValueError(arch)


def apply_freezing(model: nn.Module, cfg: ModelConfig) -> None:
    """Aplica la estrategia de transfer learning sobre `model` in-place.

    - `none`:    full fine-tuning, todos los parametros entrenables.
    - `full`:    feature extraction, backbone congelado completo; solo la cabeza.
    - `partial`: se descongelan las `partial_unfrozen_blocks` etapas finales.

    Cuando el backbone queda congelado se ponen sus capas BatchNorm en modo
    evaluacion. Si no se hiciera, las estadisticas corrientes seguirian
    actualizandose con los lotes de iNaturalist y el backbone dejaria de estar
    realmente congelado: el experimento de "feature extraction" estaria
    midiendo otra cosa.
    """
    if cfg.freeze_mode == "none":
        for param in model.parameters():
            param.requires_grad = True
        return

    head_names = set(classifier_parameter_names(model, cfg.arch))
    for name, param in model.named_parameters():
        param.requires_grad = name in head_names

    if cfg.freeze_mode == "partial":
        stages = _backbone_stages(model, cfg.arch)
        for stage in stages[-cfg.partial_unfrozen_blocks :]:
            for param in stage.parameters():
                param.requires_grad = True


def set_frozen_bn_eval(model: nn.Module, cfg: ModelConfig) -> None:
    """Fuerza `eval()` en las BatchNorm congeladas.

    Debe llamarse despues de `model.train()` en cada epoca, porque `train()`
    reactiva todos los submodulos.
    """
    if cfg.freeze_mode == "none" or not cfg.freeze_bn_stats:
        return
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            # Una BN se considera congelada si sus propios parametros afines lo estan.
            frozen = all(not p.requires_grad for p in module.parameters(recurse=False))
            if frozen:
                module.eval()


def build_model(cfg: ModelConfig, num_classes: int, image_size: int) -> nn.Module:
    """Construye el modelo descrito por `cfg` y aplica el congelamiento."""
    if cfg.arch == "mlp":
        return MLPBaseline(image_size, num_classes, cfg.mlp_hidden, cfg.dropout, cfg.batch_norm)

    if cfg.arch not in ARCHS:
        raise ValueError(f"Arquitectura no soportada: {cfg.arch}")
    constructor, weights = ARCHS[cfg.arch]
    model = constructor(weights=weights if cfg.pretrained else None)
    model = _replace_classifier(model, cfg.arch, num_classes, cfg.dropout)
    apply_freezing(model, cfg)
    return model


# ---------------------------------------------------------------------------
# Contabilidad de costo
# ---------------------------------------------------------------------------


@dataclass
class ParamStats:
    """Conteo de parametros; base del analisis costo/rendimiento."""

    total: int
    trainable: int
    frozen: int

    @property
    def trainable_fraction(self) -> float:
        return self.trainable / self.total if self.total else 0.0


def count_parameters(model: nn.Module) -> ParamStats:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return ParamStats(total=total, trainable=trainable, frozen=total - trainable)
