"""Ejecutor de experimentos para Khipu.

Recorre la cola de experimentos de una fase, salta los que ya terminaron y
entrena los pendientes dentro de un presupuesto de tiempo. El limite de 8 horas
por job de la cuenta `postgrado` obliga a que una fase pueda atravesar varios
jobs encadenados: el estado de cada experimento se guarda por epoca y el
siguiente job lo retoma exactamente donde quedo.

Uso dentro de un job de Slurm:

    python -m inat.runner --phase transfer --budget-hours 7.5

Nunca debe ejecutarse en el nodo de acceso.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig, Paths
from .data import build_loaders, make_scenario
from .engine import fit, install_signal_handler, stop_requested
from .models import build_model
from .registry import EXTRA_PHASES, PHASES
from .seeding import seed_everything

#: Margen maximo reservado para el cierre ordenado (guardar el checkpoint de la
#: epoca en curso y escribir el estado) antes del limite de Slurm.
MAX_SAFETY_MINUTES = 12.0


def prepare_process_limits() -> None:
    """Evita que el proceso agote los descriptores de archivo.

    Un job recorre decenas de experimentos y cada uno levanta su propio
    DataLoader con workers persistentes. Con la estrategia de compartido por
    descriptor que PyTorch usa por omision, cada tensor compartido consume un
    descriptor, y los de un experimento no siempre se liberan antes de que
    arranque el siguiente: tras una veintena de corridas el proceso choca con
    `OSError: [Errno 24] Too many open files` a mitad de una epoca.

    Se actua en dos frentes: compartir por memoria compartida en lugar de por
    descriptor, y subir el limite blando hasta el duro que permita el sistema.
    """
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except (RuntimeError, AttributeError):
        pass
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            log(f"limite de descriptores elevado de {soft} a {hard}")
    except (ImportError, ValueError, OSError):
        pass  # no disponible en Windows


def shutdown_loader(loader) -> None:
    """Cierra los workers de un DataLoader sin esperar al recolector de basura.

    Con `persistent_workers=True` los procesos siguen vivos mientras exista el
    iterador. Dejar que el recolector los libere mas tarde es lo que permite
    que varios experimentos tengan workers simultaneos sin necesitarlo.
    """
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return
    try:
        iterator._shutdown_workers()
    except Exception:
        pass
    loader._iterator = None


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class Budget:
    """Reloj del job; decide si queda tiempo para otra unidad de trabajo.

    El margen de cierre se escala con el presupuesto: reservar doce minutos
    fijos es razonable en un job de siete horas y medio, pero consumiria la
    mitad de un job de verificacion de treinta minutos, que entonces no
    ejecutaria nada y pasaria en verde sin haber verificado nada.
    """

    def __init__(self, hours: float):
        self.total_minutes = hours * 60
        self.deadline = time.monotonic() + hours * 3600
        self.safety_minutes = min(MAX_SAFETY_MINUTES, max(2.0, 0.10 * self.total_minutes))

    @property
    def remaining_minutes(self) -> float:
        return (self.deadline - time.monotonic()) / 60

    @property
    def exhausted(self) -> bool:
        return self.remaining_minutes < self.safety_minutes

    def has_room_for(self, minutes: float) -> bool:
        return self.remaining_minutes - self.safety_minutes > minutes


def _taxon_vector(scenario, rank: str = "class") -> np.ndarray:
    """Grupo taxonomico por indice de clase, para las metricas desagregadas."""
    cats = scenario.categories.sort_values("label")
    if rank not in cats.columns:
        return np.array(["desconocido"] * scenario.num_classes, dtype=object)
    return cats[rank].fillna("desconocido").to_numpy(dtype=object)


def run_experiment(cfg: ExperimentConfig, paths: Paths, device: torch.device,
                   budget: Budget, scenario_cache: dict) -> bool:
    """Entrena un experimento. Devuelve True si quedo terminado."""
    out_json = paths.results / f"{cfg.run_name}.json"
    if out_json.exists():
        log(f"SKIP {cfg.run_name} (ya completado)")
        return True

    ckpt_dir = paths.results / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"{cfg.run_name}.pt"

    # Arrancar con poco tiempo no pierde trabajo: el estado se guarda por epoca
    # y el siguiente job de la cadena retoma la corrida. El umbral solo evita
    # empezar algo que no alcanzaria a completar ni una epoca.
    if not budget.has_room_for(2.0):
        log(f"DIFERIDO {cfg.run_name}: quedan {budget.remaining_minutes:.0f} min")
        return False

    log(f"RUN  {cfg.run_name}  [{cfg.group}]  {cfg.notes}")
    seed_everything(cfg.seed)

    # Los escenarios de datos son caros de construir y se repiten entre
    # experimentos; se cachean por su configuracion de datos relevante.
    key = (cfg.data.scenario, cfg.data.subset_classes, cfg.data.subset_images_per_class,
           cfg.data.longtail_seed, cfg.data.longtail_head_n, cfg.data.longtail_mid_n,
           cfg.data.longtail_tail_n, cfg.data.longtail_head_frac, cfg.data.longtail_mid_frac)
    if key not in scenario_cache:
        scenario_cache[key] = make_scenario(paths, cfg.data)
        s = scenario_cache[key]
        log(f"     escenario: {len(s.train):,} train / {len(s.val):,} val / {s.num_classes} clases")
    scenario = scenario_cache[key]

    train_loader, val_loader = build_loaders(
        scenario, cfg.data, paths, cfg.seed, weighted_sampler=cfg.optim.weighted_sampler
    )
    model = build_model(cfg.model, scenario.num_classes, cfg.data.image_size)

    try:
        result = fit(
            model, train_loader, val_loader, cfg, device, scenario.num_classes,
            scenario.class_counts, scenario.group_of_class, _taxon_vector(scenario),
            checkpoint_path=ckpt, log_fn=lambda m: print(m, flush=True),
        )
    except KeyboardInterrupt:
        log(f"     {cfg.run_name} interrumpido; el estado quedo en {ckpt.name}")
        return False
    except torch.cuda.OutOfMemoryError:
        log(f"FALLO {cfg.run_name}: memoria de GPU insuficiente")
        (paths.results / f"{cfg.run_name}.failed.json").write_text(
            json.dumps({"error": "cuda_oom", "config": cfg.to_dict()}, indent=2, default=str),
            encoding="utf-8",
        )
        torch.cuda.empty_cache()
        return True
    finally:
        # Liberar los workers de este experimento antes de que arranque el
        # siguiente, pase lo que pase: es lo que impide que los descriptores se
        # acumulen a lo largo de la cola.
        shutdown_loader(train_loader)
        shutdown_loader(val_loader)
        del train_loader, val_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_json.write_text(result.to_json(), encoding="utf-8")
    if cfg.keep_weights:
        # Pesos de la mejor epoca: los necesita el analisis de errores, que se
        # hace despues y en otra maquina.
        weights_dir = paths.results / "models"
        weights_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict(),
                    "num_classes": scenario.num_classes, "final": result.final},
                   weights_dir / f"{cfg.run_name}.pt")
        log(f"     pesos guardados en models/{cfg.run_name}.pt")
    ckpt.unlink(missing_ok=True)
    log(
        f"DONE {cfg.run_name}  top1={result.final['top1']:.4f} "
        f"top5={result.final['top5']:.4f} mF1={result.final['macro_f1']:.4f} "
        f"({result.gpu_hours:.2f} GPU-h, {result.params_trainable / 1e6:.1f}M entrenables)"
    )
    return True


def best_transfer_lr(paths: Paths) -> dict:
    """Lee las sondas de learning rate y elige el mejor lr por profundidad k.

    El patron `TP_k*_*.json` tambien casa con los `*.failed.json`, que no
    describen una corrida sino un error y no tienen `exp_id`. Ignorarlos es
    indispensable: sin ese filtro, una unica sonda fallida hace que esta
    funcion lance una excepcion cada vez que se construye la cola, y la fase
    entera se cae en lugar de continuar con las sondas que si terminaron.
    """
    best: dict[int, tuple[float, float]] = {}
    for path in sorted(paths.results.glob("TP_k*_*.json")):
        if path.name.endswith(".failed.json"):
            continue
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            k = int(blob["exp_id"].split("_")[1][1:])
            lr = float(blob["config"]["optim"]["lr"])
            score = blob["final"]["top1"]
        except (KeyError, ValueError, IndexError, json.JSONDecodeError) as exc:
            log(f"  sonda ilegible, se ignora: {path.name} ({exc})")
            continue
        if k not in best or score > best[k][1]:
            best[k] = (lr, score)
    return {k: v[0] for k, v in best.items()}


def select_only(queue: list[ExperimentConfig], only: str | None) -> list[ExperimentConfig]:
    """Deja solo las corridas cuyo id o nombre empieza por alguno de los prefijos.

    Sirve para repartir la cola entre varias GPU. Cada proceso recibe una lista
    distinta y escribe en la misma carpeta de resultados; como los nombres no
    se solapan, no hay dos procesos entrenando la misma corrida.
    """
    if not only:
        return queue
    keys = [key.strip() for key in only.split(",") if key.strip()]
    return [
        cfg for cfg in queue
        if any(cfg.exp_id == key or cfg.run_name.startswith(key) for key in keys)
    ]


def build_queue(phase: str, paths: Paths) -> list[ExperimentConfig]:
    """Construye la cola de la fase, inyectando resultados de fases previas."""
    if phase == "transfer":
        chosen = best_transfer_lr(paths)
        if chosen:
            log(f"learning rates elegidos por las sondas: {chosen}")
        return PHASES["transfer"](chosen or None)
    if phase == "longtail":
        chosen = best_transfer_lr(paths)
        return PHASES["longtail"](chosen.get(2, 1e-3))
    if phase in EXTRA_PHASES:
        return EXTRA_PHASES[phase]()
    if phase not in PHASES:
        known = ", ".join([*PHASES, *EXTRA_PHASES, "all"])
        raise SystemExit(f"Fase desconocida: {phase}. Opciones: {known}")
    return PHASES[phase]()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, help="fase a ejecutar, o 'all'")
    parser.add_argument("--budget-hours", type=float, default=7.5,
                        help="tiempo de calculo antes de cerrar ordenadamente")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="ejecuta a lo sumo N experimentos (depuracion)")
    parser.add_argument("--dry-run", action="store_true",
                        help="imprime la cola sin entrenar")
    parser.add_argument("--only", default=None,
                        help="prefijos de exp_id o run_name, separados por comas")
    args = parser.parse_args(argv)

    paths = Paths(
        data_root=args.data_root or Paths().data_root,
        results=args.results or Paths().results,
    )
    paths.results.mkdir(parents=True, exist_ok=True)
    install_signal_handler()
    prepare_process_limits()

    # Con 'all' las fases se recorren en el orden declarado en PHASES, y la
    # cola de cada una se construye justo antes de ejecutarla: la fase
    # 'transfer' necesita leer los resultados de 'transfer_lr', que pueden
    # haberse producido en este mismo job.
    phase_names = list(PHASES) if args.phase == "all" else [args.phase]

    if args.dry_run:
        total = 0
        for name in phase_names:
            queue = select_only(build_queue(name, paths), args.only)
            print(f"\n--- {name} ({len(queue)}) ---")
            for cfg in queue:
                done = (paths.results / f"{cfg.run_name}.json").exists()
                print(f"{'[hecho] ' if done else '[pend.] '}{cfg.run_name:<44} {cfg.notes}")
            total += len(queue)
        print(f"\n{total} experimentos en total")
        return 0

    if not torch.cuda.is_available():
        log("ADVERTENCIA: no hay GPU disponible; la corrida sera inviablemente lenta")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}  "
            f"({torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB)")
        # TF32 acelera las convoluciones y el matmul de la cabeza de 10,000
        # clases sin efecto medible en la exactitud final.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    budget = Budget(args.budget_hours)
    scenario_cache: dict = {}
    completed = deferred = pending = queue_size = 0
    out_of_time = False

    for name in phase_names:
        if out_of_time or stop_requested():
            break
        queue = select_only(build_queue(name, paths), args.only)
        if args.limit:
            queue = queue[: args.limit]
        queue_size += len(queue)
        log(f"=== fase {name}: {len(queue)} experimentos ===")

        for cfg in queue:
            if stop_requested():
                log("senal de Slurm recibida; se detiene la cola")
                out_of_time = True
                break
            try:
                if run_experiment(cfg, paths, device, budget, scenario_cache):
                    completed += 1
                else:
                    deferred += 1
                    if budget.exhausted:
                        out_of_time = True
                        break
            except Exception:  # un experimento roto no debe tumbar la fase entera
                log(f"ERROR en {cfg.run_name}:\n{traceback.format_exc()}")
                (paths.results / f"{cfg.run_name}.failed.json").write_text(
                    json.dumps({"error": traceback.format_exc(), "config": cfg.to_dict()},
                               indent=2, default=str), encoding="utf-8")
        pending += sum(
            1 for c in queue if not (paths.results / f"{c.run_name}.json").exists()
        )

    status = {
        "phase": args.phase, "completed_this_job": completed, "deferred": deferred,
        "pending_total": pending, "queue_size": queue_size,
        "remaining_minutes": round(budget.remaining_minutes, 1),
        "safety_minutes": round(budget.safety_minutes, 1),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (paths.results / f"status_{args.phase}.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8")
    log(f"fin de job: {json.dumps(status)}")
    # Exit 0 aunque queden pendientes: la cadena `afterok` debe continuar para
    # que el siguiente job retome la cola.
    return 0


if __name__ == "__main__":
    sys.exit(main())
