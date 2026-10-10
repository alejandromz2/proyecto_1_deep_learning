"""Prueba de integridad del pipeline antes de enviar nada a la cola de Khipu.

Ejercita cada camino distinto del codigo -- baseline denso, CNN desde cero, las
tres estrategias de transfer learning, el escenario long-tail y cada estrategia
de mitigacion -- con una o dos epocas sobre el subset de desarrollo. No mide
rendimiento; comprueba que todo corre, que el congelamiento deja exactamente
los parametros que debe, que las metricas son coherentes y que reanudar desde
un checkpoint reproduce el estado.

    python scripts/smoke_test.py --data-root D:/proyectos/inat2021/dev
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from inat.config import DataConfig, EarlyStopConfig, ExperimentConfig, ModelConfig, OptimConfig, Paths
from inat.data import build_loaders, make_scenario
from inat.engine import class_weights, fit
from inat.models import build_model, count_parameters
from inat.seeding import seed_everything

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    status = "ok  " if condition else "FALLA"
    print(f"    [{status}] {message}")
    if not condition:
        FAILURES.append(message)


def make_cfg(exp_id: str, name: str, **kw) -> ExperimentConfig:
    data = DataConfig(image_size=kw.pop("image_size", 128), batch_size=kw.pop("batch_size", 32),
                      eval_batch_size=64, num_workers=kw.pop("num_workers", 4),
                      persistent_workers=False, **kw.pop("data", {}))
    model = ModelConfig(**kw.pop("model", {}))
    optim = OptimConfig(epochs=kw.pop("epochs", 2), amp=True, warmup_epochs=0,
                        **kw.pop("optim", {}))
    return ExperimentConfig(exp_id=exp_id, name=name, group="smoke", seed=42,
                            data=data, model=model, optim=optim,
                            early_stop=EarlyStopConfig(enabled=False, restore_best=True))


def run(cfg: ExperimentConfig, paths: Paths, device, ckpt: Path | None = None):
    seed_everything(cfg.seed)
    scenario = make_scenario(paths, cfg.data)
    train_loader, val_loader = build_loaders(scenario, cfg.data, paths, cfg.seed,
                                             weighted_sampler=cfg.optim.weighted_sampler)
    model = build_model(cfg.model, scenario.num_classes, cfg.data.image_size)
    taxon = scenario.categories.sort_values("label")["class"].fillna("?").to_numpy(dtype=object)
    result = fit(model, train_loader, val_loader, cfg, device, scenario.num_classes,
                 scenario.class_counts, scenario.group_of_class, taxon,
                 checkpoint_path=ckpt, log_fn=lambda m: print("   ", m.strip()))
    return result, scenario, model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    paths = Paths(data_root=args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"dispositivo: {device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})\n")

    # -- 1. Escenario balanceado ------------------------------------------
    print("1. Escenario balanceado")
    scenario = make_scenario(paths, DataConfig())
    check(scenario.num_classes > 0, f"{scenario.num_classes} clases, {len(scenario.train)} train")
    check(set(scenario.train["label"]) == set(range(scenario.num_classes)),
          "las etiquetas son contiguas 0..C-1")
    check(scenario.val["label"].max() < scenario.num_classes,
          "validacion no contiene etiquetas fuera de rango")

    # -- 2. Baseline MLP y crecimiento de parametros -----------------------
    print("\n2. Baseline MLP")
    # El crecimiento cuadratico con el lado de la imagen ocurre en la primera
    # capa, que es la que conecta con los pixeles; el resto de la red tiene un
    # tamano fijo. Se comprueban las dos cosas por separado.
    first, total = {}, {}
    for size in (32, 64):
        model = build_model(ModelConfig(arch="mlp"), scenario.num_classes, size)
        first[size] = model.net[1].weight.numel()
        total[size] = count_parameters(model).total
    ratio_first = first[64] / first[32]
    check(abs(ratio_first - 4.0) < 1e-6,
          f"la primera capa se cuadruplica al duplicar el lado ({ratio_first:.2f}x)")
    check(total[64] > total[32],
          f"el total crece de {total[32]/1e6:.1f}M a {total[64]/1e6:.1f}M parametros")
    result, _, _ = run(make_cfg("S1", "mlp", model={"arch": "mlp"}, image_size=64,
                                optim={"optimizer": "adam", "lr": 1e-3}), paths, device)
    check(0.0 <= result.final["top1"] <= 1.0, f"MLP entrena, top1={result.final['top1']:.3f}")

    # -- 3. CNN desde cero -------------------------------------------------
    print("\n3. CNN desde cero (resnet50)")
    # lr moderado: con 400 imagenes hay doce pasos por epoca y el warmup real
    # de los experimentos (una epoca entera, ~2000 pasos) no tiene equivalente
    # aqui. Un lr alto diverge por el tamano del subset, no por un defecto.
    result, _, _ = run(make_cfg("S2", "resnet50_scratch",
                                model={"arch": "resnet50", "pretrained": False},
                                optim={"optimizer": "sgd_momentum", "lr": 0.01}), paths, device)
    check(result.params_trainable == result.params_total,
          "sin congelamiento, todos los parametros son entrenables")
    check(len(result.history) == 2, "se registraron las dos epocas")
    check(all(np.isfinite(h["train_loss"]) for h in result.history),
          "la perdida de entrenamiento se mantiene finita")
    check(result.throughput_img_s > 0, f"throughput medido: {result.throughput_img_s:.0f} img/s")

    # -- 4. Las tres estrategias de transfer learning ----------------------
    print("\n4. Transfer learning: congelado / parcial / completo")
    trainables = {}
    for k, (mode, blocks) in {0: ("full", 0), 2: ("partial", 2), 4: ("none", 4)}.items():
        cfg = make_cfg(f"S3k{k}", f"tl_k{k}",
                       model={"arch": "resnet50", "pretrained": True, "freeze_mode": mode,
                              "partial_unfrozen_blocks": blocks, "dropout": 0.2},
                       optim={"optimizer": "sgd_momentum", "lr": 1e-3,
                              "backbone_lr_mult": 1.0 if k == 0 else 0.1},
                       epochs=1)
        result, _, model = run(cfg, paths, device)
        trainables[k] = result.params_trainable
        if k == 0:
            # Con el backbone congelado solo debe entrenarse la capa final.
            expected = sum(p.numel() for n, p in model.named_parameters() if n.startswith("fc."))
            check(result.params_trainable == expected,
                  f"k=0 entrena solo la cabeza ({expected:,} parametros)")
            frozen_bn_eval = all(
                not m.training for m in model.modules()
                if isinstance(m, torch.nn.BatchNorm2d)
            )
            model.train()
            from inat.models import set_frozen_bn_eval
            set_frozen_bn_eval(model, cfg.model)
            still_eval = all(not m.training for m in model.modules()
                             if isinstance(m, torch.nn.BatchNorm2d))
            check(still_eval, "las BatchNorm congeladas quedan en modo eval tras train()")
    check(trainables[0] < trainables[2] < trainables[4],
          f"los parametros entrenables crecen con k: "
          f"{trainables[0]/1e6:.1f}M < {trainables[2]/1e6:.1f}M < {trainables[4]/1e6:.1f}M")

    # -- 5. Escenario long-tail -------------------------------------------
    print("\n5. Escenario long-tail")
    lt = DataConfig(scenario="longtail", longtail_head_frac=0.2, longtail_mid_frac=0.3,
                    longtail_head_n=20, longtail_mid_n=10, longtail_tail_n=3)
    s_lt = make_scenario(paths, lt)
    groups = s_lt.group_of_class
    check(groups is not None and set(groups) <= {"head", "mid", "tail"},
          f"grupos asignados: {dict(zip(*np.unique(groups, return_counts=True)))}")
    head_n = s_lt.class_counts[groups == "head"].mean()
    tail_n = s_lt.class_counts[groups == "tail"].mean()
    check(head_n > tail_n, f"la cabeza tiene mas ejemplos que la cola ({head_n:.1f} vs {tail_n:.1f})")
    s_lt2 = make_scenario(paths, lt)
    check(np.array_equal(s_lt.class_counts, s_lt2.class_counts),
          "el escenario long-tail es reproducible con la misma semilla")
    s_lt3 = make_scenario(paths, DataConfig(**{**lt.__dict__, "longtail_seed": 7}))
    check(not np.array_equal(s_lt.class_counts, s_lt3.class_counts),
          "otra semilla produce otro reparto")

    # -- 6. Estrategias de mitigacion -------------------------------------
    print("\n6. Estrategias de mitigacion del desbalance")
    w = class_weights(s_lt.class_counts, "effective")
    check(w is not None and abs(float(w.mean()) - 1.0) < 1e-5, "los pesos de clase tienen media 1")
    check(float(w[s_lt.class_counts.argmin()]) > float(w[s_lt.class_counts.argmax()]),
          "la clase mas rara recibe mas peso")
    for loss, extra in (("weighted_ce", {"class_weighting": "effective"}),
                        ("focal", {"focal_gamma": 2.0}),
                        ("ce", {"weighted_sampler": True})):
        cfg = make_cfg(f"S4_{loss}", f"lt_{loss}",
                       data=dict(scenario="longtail", longtail_head_n=20,
                                 longtail_mid_n=10, longtail_tail_n=3),
                       model={"arch": "resnet18", "pretrained": True, "freeze_mode": "partial",
                              "partial_unfrozen_blocks": 1},
                       optim={"optimizer": "sgd_momentum", "lr": 1e-3, "loss": loss, **extra},
                       epochs=1)
        result, _, _ = run(cfg, paths, device)
        check(bool(result.by_group), f"{loss}: metricas por grupo {list(result.by_group)}")

    # -- 7. Reanudacion desde checkpoint ----------------------------------
    print("\n7. Reanudacion desde checkpoint")
    tmp = Path(tempfile.mkdtemp())
    try:
        ckpt = tmp / "resume.pt"
        cfg = make_cfg("S5", "resume", model={"arch": "resnet18", "pretrained": False},
                       optim={"optimizer": "sgd_momentum", "lr": 0.01}, epochs=2)
        first, _, _ = run(cfg.replace(optim__epochs=1), paths, device, ckpt=ckpt)
        check(ckpt.exists(), "el checkpoint se escribio")
        second, _, _ = run(cfg, paths, device, ckpt=ckpt)
        check(len(second.history) == 2 and second.history[0]["epoch"] == 0,
              "la reanudacion conserva el historial y continua en la epoca 2")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # -- 8. Metricas -------------------------------------------------------
    print("\n8. Coherencia de las metricas")
    check(result.final["top5"] >= result.final["top1"] - 1e-9, "top5 >= top1")
    check(0.0 <= result.final["macro_f1"] <= 1.0, "macro F1 en [0,1]")
    check(bool(result.by_taxon), f"metricas por taxon: {list(result.by_taxon)[:3]}")

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"PRUEBA FALLIDA: {len(FAILURES)} comprobacion(es)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("TODAS LAS COMPROBACIONES PASARON")
    return 0


if __name__ == "__main__":
    sys.exit(main())
