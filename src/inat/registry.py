"""Diseno experimental: definicion declarativa de todas las corridas.

Cada funcion devuelve la lista de `ExperimentConfig` de un bloque del informe.
Mantener el diseno aqui, y no disperso en el notebook, es lo que permite
afirmar que dos experimentos son comparables: las condiciones compartidas se
escriben una sola vez y cada experimento declara explicitamente que variable
cambia respecto del control.

Convencion de identificadores:
    B*  baseline MLP            O*  optimizacion
    A*  arquitecturas CNN       R*  regularizacion y ablation
    T*  transfer learning       L*  long-tail
"""
from __future__ import annotations

from .config import DataConfig, EarlyStopConfig, ExperimentConfig, ModelConfig, OptimConfig

# --------------------------------------------------------------------------
# Condiciones compartidas
# --------------------------------------------------------------------------

#: Resolucion de la grilla exploratoria. 160 px mantiene el problema
#: fine-grained reconocible y permite recorrer ~20 configuraciones en el
#: presupuesto de GPU disponible.
GRID_SIZE = 160

#: Resolucion del bloque de transfer learning. Se usa 224 px porque es la
#: resolucion con la que se entrenaron los pesos de ImageNet: evaluar un
#: backbone preentrenado fuera de su escala nativa penalizaria precisamente la
#: condicion que el experimento quiere medir.
TL_SIZE = 224

GRID_EPOCHS = 12
TL_EPOCHS = 15
PROBE_EPOCHS = 3   # corridas cortas para barridos de learning rate

SEEDS = (42, 43)


def _data(**kwargs) -> DataConfig:
    base = dict(image_size=GRID_SIZE, batch_size=256, num_workers=12)
    base.update(kwargs)
    return DataConfig(**base)


def _optim(**kwargs) -> OptimConfig:
    base = dict(
        optimizer="sgd_momentum", lr=0.1, momentum=0.9, weight_decay=5e-5,
        scheduler="cosine", warmup_epochs=1, epochs=GRID_EPOCHS, amp=True,
    )
    base.update(kwargs)
    return OptimConfig(**base)


def _es(**kwargs) -> EarlyStopConfig:
    base = dict(enabled=True, metric="val_top1", mode="max", patience=4, restore_best=True)
    base.update(kwargs)
    return EarlyStopConfig(**base)


# --------------------------------------------------------------------------
# Parte 2 -- Baseline con MLP
# --------------------------------------------------------------------------


def baseline_experiments() -> list[ExperimentConfig]:
    """Baseline completamente conectado y efecto de la resolucion de entrada.

    El MLP no busca competir: sirve para medir cuanto se pierde al descartar la
    estructura espacial. El barrido de resolucion hace visible el otro problema
    del enfoque, que el numero de parametros de la primera capa crece con el
    cuadrado del lado de la imagen.
    """
    exps = []
    for i, size in enumerate((32, 48, 64)):
        exps.append(
            ExperimentConfig(
                exp_id=f"B{i + 1}",
                name=f"mlp_{size}px",
                group="baseline",
                seed=SEEDS[0],
                data=_data(image_size=size, batch_size=512, random_resized_crop=False,
                           horizontal_flip=True),
                model=ModelConfig(arch="mlp", mlp_hidden=(2048, 1024), dropout=0.2,
                                  batch_norm=True),
                # Adam en lugar de SGD: una red densa sobre pixeles crudos tiene
                # gradientes muy mal escalados entre capas y SGD con un unico lr
                # global converge mucho peor en las pruebas preliminares.
                optim=_optim(optimizer="adam", lr=1e-3, weight_decay=1e-4, epochs=10,
                             warmup_epochs=0),
                early_stop=_es(patience=3),
                notes="Baseline MLP; varia solo la resolucion de entrada.",
            )
        )
    # Ablation de BatchNorm aislada en el MLP, donde si se controla la
    # arquitectura (las CNN de torchvision la traen incrustada).
    exps.append(
        exps[-1].replace(
            exp_id="B4", name="mlp_64px_sin_bn", model__batch_norm=False,
            notes="Ablation: identico a B3 sin BatchNorm.",
        )
    )
    return exps


# --------------------------------------------------------------------------
# Parte 3 -- Arquitecturas CNN modernas
# --------------------------------------------------------------------------


def architecture_experiments() -> list[ExperimentConfig]:
    """Comparacion de CNN modernas entrenadas desde inicializacion aleatoria.

    Mismo presupuesto de epocas, misma resolucion, mismo optimizador y mismo
    augmentation: la unica variable es la arquitectura. ResNet-50 aporta la
    referencia residual clasica, EfficientNet-B0 el escalado compuesto con
    bloques invertidos, y MobileNetV3-Large el extremo barato de la curva de
    costo, necesario para que la comparacion tenga un eje de eficiencia y no
    solo de exactitud.
    """
    archs = ["resnet50", "efficientnet_b0", "mobilenet_v3_large"]
    return [
        ExperimentConfig(
            exp_id=f"A{i + 1}",
            name=f"{arch}_scratch",
            group="arquitectura",
            seed=SEEDS[0],
            data=_data(color_jitter=0.0),
            model=ModelConfig(arch=arch, pretrained=False),
            optim=_optim(),
            early_stop=_es(),
            notes="Desde cero; condiciones identicas entre arquitecturas.",
        )
        for i, arch in enumerate(archs)
    ]


# --------------------------------------------------------------------------
# Parte 4 -- Optimizacion
# --------------------------------------------------------------------------


def optimizer_experiments() -> list[ExperimentConfig]:
    """Comparacion de optimizadores sobre la arquitectura de referencia.

    Cada optimizador se evalua en su learning rate habitual, no en uno comun:
    comparar SGD y Adam con el mismo lr mide sobre todo la escala del paso
    efectivo, no el algoritmo. El barrido corto de lr que acompana al bloque
    documenta esa sensibilidad de forma explicita.
    """
    setups = [
        ("sgd", 0.1), ("sgd_momentum", 0.1), ("adam", 1e-3),
        ("adamw", 1e-3), ("rmsprop", 1e-3),
    ]
    exps = [
        ExperimentConfig(
            exp_id=f"O{i + 1}",
            name=f"resnet50_{opt}",
            group="optimizacion",
            seed=SEEDS[0],
            data=_data(),
            model=ModelConfig(arch="resnet50", pretrained=False),
            optim=_optim(optimizer=opt, lr=lr),
            early_stop=_es(),
            notes="Solo cambia el optimizador (y su lr de referencia).",
        )
        for i, (opt, lr) in enumerate(setups)
    ]

    # Sensibilidad al learning rate: corridas cortas, suficientes para ver
    # divergencia o estancamiento sin gastar el presupuesto completo.
    probes = []
    for opt, lrs in (("sgd_momentum", (0.01, 0.1, 0.4)), ("adam", (1e-4, 1e-3, 5e-3))):
        for lr in lrs:
            probes.append(
                ExperimentConfig(
                    exp_id=f"OS_{opt}_{lr:g}",
                    name=f"lrprobe_{opt}_{lr:g}",
                    group="optimizacion_lr",
                    seed=SEEDS[0],
                    data=_data(),
                    model=ModelConfig(arch="resnet50", pretrained=False),
                    optim=_optim(optimizer=opt, lr=lr, epochs=PROBE_EPOCHS, warmup_epochs=0),
                    early_stop=_es(enabled=False),
                    notes="Sonda corta de sensibilidad al learning rate.",
                )
            )
    return exps + probes


# --------------------------------------------------------------------------
# Parte 5 -- Regularizacion, augmentation y ablation study
# --------------------------------------------------------------------------


def regularization_experiments() -> list[ExperimentConfig]:
    """Ablation incremental sobre ResNet-50 desde cero.

    La escalera R1->R4 anade un componente por vez, de modo que la diferencia
    entre filas consecutivas sea atribuible a ese componente. R5 y R6 aislan
    dos transformaciones de augmentation cuyo efecto no es obvio en un problema
    fine-grained: el color jitter fuerte puede destruir precisamente el rasgo
    que separa dos especies proximas, y el random erasing puede ocultar la
    region diagnostica cuando el organismo ocupa poca area de la imagen.
    """
    base_data = dict(random_resized_crop=False, horizontal_flip=False)
    ladder = [
        ("R1", "sin_regularizacion", dict(**base_data), 0.0, 0.0),
        ("R2", "mas_augmentation", dict(random_resized_crop=True, horizontal_flip=True), 0.0, 0.0),
        ("R3", "mas_dropout", dict(random_resized_crop=True, horizontal_flip=True), 0.3, 0.0),
        ("R4", "mas_weight_decay", dict(random_resized_crop=True, horizontal_flip=True), 0.3, 5e-4),
    ]
    exps = [
        ExperimentConfig(
            exp_id=eid, name=name, group="regularizacion", seed=SEEDS[0],
            data=_data(**data_kwargs),
            model=ModelConfig(arch="resnet50", pretrained=False, dropout=dropout),
            optim=_optim(weight_decay=wd),
            early_stop=_es(),
            notes="Ablation incremental; cada fila anade un componente.",
        )
        for eid, name, data_kwargs, dropout, wd in ladder
    ]

    strong = dict(random_resized_crop=True, horizontal_flip=True)
    exps.append(
        ExperimentConfig(
            exp_id="R5", name="color_jitter_fuerte", group="regularizacion", seed=SEEDS[0],
            data=_data(color_jitter=0.4, **strong),
            model=ModelConfig(arch="resnet50", pretrained=False, dropout=0.3),
            optim=_optim(weight_decay=5e-4), early_stop=_es(),
            notes="R4 + color jitter fuerte; prueba si el color es rasgo diagnostico.",
        )
    )
    exps.append(
        ExperimentConfig(
            exp_id="R6", name="random_erasing", group="regularizacion", seed=SEEDS[0],
            data=_data(random_erasing=0.25, **strong),
            model=ModelConfig(arch="resnet50", pretrained=False, dropout=0.3),
            optim=_optim(weight_decay=5e-4), early_stop=_es(),
            notes="R4 + random erasing.",
        )
    )
    return exps


# --------------------------------------------------------------------------
# Parte 6 -- Transfer Learning y Fine-Tuning  (bloque principal)
# --------------------------------------------------------------------------

#: Learning rates candidatos por profundidad de descongelamiento. Comparar las
#: cinco condiciones con un unico lr seria un experimento confundido: con el
#: backbone congelado solo se entrena una capa lineal y tolera (necesita) un
#: paso mayor, mientras que en fine-tuning completo ese mismo paso degrada las
#: representaciones de ImageNet. Cada condicion se mide en su mejor lr.
TL_LR_GRID = {
    0: (3e-3, 1e-2, 3e-2),
    1: (3e-4, 1e-3, 3e-3),
    2: (3e-4, 1e-3, 3e-3),
    3: (1e-4, 3e-4, 1e-3),
    4: (1e-4, 3e-4, 1e-3),
}


def _tl_config(k: int, lr: float, seed: int, epochs: int, exp_id: str, name: str,
               group: str, pretrained: bool = True) -> ExperimentConfig:
    """Una condicion del barrido de profundidad de descongelamiento.

    `k` es el numero de etapas finales del backbone que se entrenan: k=0 es
    feature extraction con el backbone totalmente congelado, k=4 es fine-tuning
    completo, y los valores intermedios son fine-tuning parcial.
    """
    if k == 0:
        freeze_mode, blocks = "full", 0
    elif k >= 4:
        freeze_mode, blocks = "none", 4
    else:
        freeze_mode, blocks = "partial", k

    return ExperimentConfig(
        exp_id=exp_id, name=name, group=group, seed=seed,
        data=_data(image_size=TL_SIZE, batch_size=192, random_resized_crop=True,
                   horizontal_flip=True),
        model=ModelConfig(arch="resnet50", pretrained=pretrained, dropout=0.2,
                          freeze_mode=freeze_mode, partial_unfrozen_blocks=blocks,
                          freeze_bn_stats=True),
        # lr diferencial: el backbone preentrenado avanza diez veces mas lento
        # que la cabeza nueva, que parte de pesos aleatorios.
        optim=_optim(optimizer="sgd_momentum", lr=lr,
                     backbone_lr_mult=1.0 if k == 0 else 0.1,
                     weight_decay=1e-4, epochs=epochs, warmup_epochs=1),
        early_stop=_es(patience=4),
        notes=f"Transfer learning con k={k} etapas finales descongeladas.",
        # Solo las corridas largas del bloque principal conservan pesos; las
        # sondas de learning rate no se analizan despues.
        keep_weights=epochs >= TL_EPOCHS,
    )


def transfer_lr_probes() -> list[ExperimentConfig]:
    """Barrido corto de learning rate por condicion, previo al bloque largo."""
    out = []
    for k, lrs in TL_LR_GRID.items():
        for lr in lrs:
            cfg = _tl_config(k, lr, SEEDS[0], PROBE_EPOCHS,
                             exp_id=f"TP_k{k}_{lr:g}", name=f"tlprobe_k{k}_{lr:g}",
                             group="transfer_lr")
            out.append(cfg.replace(early_stop__enabled=False))
    return out


def transfer_experiments(best_lr: dict | None = None) -> list[ExperimentConfig]:
    """Bloque principal: barrido de profundidad de descongelamiento k = 0..4.

    Responde las dos preguntas de investigacion del proyecto con un unico
    experimento controlado. k=0 (backbone congelado) y k=4 (fine-tuning
    completo) son los extremos que la primera pregunta compara; los valores
    intermedios trazan la frontera rendimiento/costo que la segunda pregunta
    requiere, en lugar de inferirla de un unico punto.

    Se anade un control entrenado desde cero con el mismo presupuesto, sin el
    cual no es posible separar el aporte del preentrenamiento del aporte de la
    arquitectura.

    `best_lr` mapea k -> learning rate elegido por las sondas. Se completa
    profundidad a profundidad con el valor central de la grilla: si una sonda
    concreta no llego a terminar, esa condicion usa el valor por omision y el
    barrido se ejecuta igual, en lugar de caerse entero por una pieza ausente.
    """
    defaults = {k: lrs[1] for k, lrs in TL_LR_GRID.items()}
    best_lr = {**defaults, **(best_lr or {})}
    labels = {0: "frozen", 1: "partial_k1", 2: "partial_k2", 3: "partial_k3", 4: "full_ft"}

    exps = []
    for k in range(5):
        # Semilla adicional en los tres puntos sobre los que se apoyan las
        # conclusiones (congelado, parcial intermedio y completo), para poder
        # reportar dispersion donde realmente importa.
        seeds = SEEDS if k in (0, 2, 4) else SEEDS[:1]
        for seed in seeds:
            exps.append(
                _tl_config(k, best_lr[k], seed, TL_EPOCHS,
                           exp_id=f"T{k}", name=f"resnet50_{labels[k]}",
                           group="transfer")
            )

    exps.append(
        _tl_config(4, 0.1, SEEDS[0], TL_EPOCHS, exp_id="T5",
                   name="resnet50_scratch_224", group="transfer", pretrained=False)
        .replace(optim__backbone_lr_mult=1.0,
                 notes="Control sin preentrenamiento, mismo presupuesto que T4.")
    )
    return exps


# --------------------------------------------------------------------------
# Parte 7 -- Long-Tail Challenge
# --------------------------------------------------------------------------

#: Cuotas del escenario desbalanceado. Con 10,000 especies resultan ~185,000
#: imagenes y una razon de desbalance de 10:1 entre la cabeza y la cola.
LT_QUOTAS = dict(longtail_head_frac=0.20, longtail_mid_frac=0.30,
                 longtail_head_n=50, longtail_mid_n=20, longtail_tail_n=5)

#: Imagenes por clase del control balanceado de igual tamano total. Permite
#: separar el efecto del desbalance del efecto de "hay menos datos".
LT_MATCHED_N = 18


def longtail_experiments(best_lr: float = 1e-3) -> list[ExperimentConfig]:
    """Escenario long-tail y estrategias de mitigacion.

    L1 y L2 son los dos controles necesarios: el corpus balanceado completo y
    un corpus balanceado con el mismo numero total de imagenes que el
    desbalanceado. Sin L2, cualquier caida observada en L3 se podria atribuir
    al desbalance cuando en realidad proviene de entrenar con un tercio de los
    datos. L4-L6 evaluan tres mitigaciones de naturaleza distinta: reponderar
    la perdida, reponderar el muestreo, y reponderar por dificultad.
    """
    def base(exp_id, name, data_kwargs, optim_kwargs, notes):
        return ExperimentConfig(
            exp_id=exp_id, name=name, group="longtail", seed=SEEDS[0],
            data=_data(image_size=GRID_SIZE, batch_size=192, **data_kwargs),
            model=ModelConfig(arch="resnet50", pretrained=True, dropout=0.2,
                              freeze_mode="partial", partial_unfrozen_blocks=2),
            optim=_optim(optimizer="sgd_momentum", lr=best_lr, backbone_lr_mult=0.1,
                         weight_decay=1e-4, epochs=GRID_EPOCHS, **optim_kwargs),
            early_stop=_es(metric="val_macro_f1"),
            notes=notes,
            keep_weights=True,
        )

    balanced = dict(scenario="balanced")
    lt = dict(scenario="longtail", **LT_QUOTAS)

    return [
        base("L1", "balanceado_completo", balanced, {},
             "Control: corpus balanceado completo (50 img/especie)."),
        base("L2", "balanceado_equiparado", dict(scenario="balanced",
             subset_images_per_class=LT_MATCHED_N), {},
             "Control: balanceado con el mismo total de imagenes que L3."),
        base("L3", "longtail_sin_correccion", lt, {},
             "Long-tail sin ninguna estrategia de mitigacion."),
        base("L4", "longtail_weighted_ce", lt,
             dict(loss="weighted_ce", class_weighting="effective"),
             "Mitigacion: cross-entropy ponderada por numero efectivo de muestras."),
        base("L5", "longtail_weighted_sampler", lt, dict(weighted_sampler=True),
             "Mitigacion: muestreo balanceado por clase."),
        base("L6", "longtail_focal", lt, dict(loss="focal", focal_gamma=2.0),
             "Mitigacion: focal loss."),
    ]


# --------------------------------------------------------------------------
# Catalogo
# --------------------------------------------------------------------------

def verify_experiments() -> list[ExperimentConfig]:
    """Comprobacion previa sobre el corpus real, en una GPU MIG de 5 GB.

    Recorre los tres caminos que pueden romperse al cambiar de maquina -- el
    baseline denso, una CNN desde cero y el transfer learning con backbone
    congelado -- usando 50 especies y una epoca. Su proposito es que un error
    de rutas, de pesos preentrenados o de memoria aparezca en treinta minutos y
    no ocho horas despues, con la cadena de jobs ya lanzada.
    """
    probe = dict(subset_classes=50, num_workers=6, batch_size=64)
    common = dict(group="verificacion", seed=SEEDS[0])
    return [
        ExperimentConfig(
            exp_id="V1", name="verify_mlp", **common,
            data=_data(image_size=64, **probe),
            model=ModelConfig(arch="mlp"),
            optim=_optim(optimizer="adam", lr=1e-3, epochs=1, warmup_epochs=0),
            early_stop=_es(enabled=False),
            notes="Verificacion: baseline denso sobre el corpus real.",
        ),
        ExperimentConfig(
            exp_id="V2", name="verify_cnn_scratch", **common,
            data=_data(image_size=128, **probe),
            model=ModelConfig(arch="resnet50", pretrained=False),
            optim=_optim(optimizer="sgd_momentum", lr=0.01, epochs=1, warmup_epochs=0),
            early_stop=_es(enabled=False),
            notes="Verificacion: CNN desde cero.",
        ),
        ExperimentConfig(
            exp_id="V3", name="verify_transfer", **common,
            data=_data(image_size=128, **probe),
            model=ModelConfig(arch="resnet50", pretrained=True, freeze_mode="full"),
            optim=_optim(optimizer="sgd_momentum", lr=1e-2, epochs=1, warmup_epochs=0),
            early_stop=_es(enabled=False),
            notes="Verificacion: pesos de ImageNet en cache y backbone congelado.",
        ),
    ]


#: Fases en el orden en que deben ejecutarse. Los barridos de lr preceden a los
#: bloques largos que consumen su resultado.
PHASES = {
    "baseline": baseline_experiments,
    "arquitectura": architecture_experiments,
    "optimizacion": optimizer_experiments,
    "regularizacion": regularization_experiments,
    "transfer_lr": transfer_lr_probes,
    "transfer": transfer_experiments,
    "longtail": longtail_experiments,
}


#: La verificacion no forma parte de `all`: es una compuerta previa, no un
#: experimento del informe.
EXTRA_PHASES = {"verify": verify_experiments}


def all_experiments() -> list[ExperimentConfig]:
    out: list[ExperimentConfig] = []
    for build in PHASES.values():
        out.extend(build())
    return out
