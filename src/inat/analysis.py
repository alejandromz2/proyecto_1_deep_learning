"""Agregacion de resultados: de los JSON por corrida a las tablas del informe.

El runner deja un JSON por experimento. Este modulo los lee y produce las
vistas que el notebook muestra: la tabla final de experimentos, las curvas de
aprendizaje, el resumen costo/rendimiento del bloque de transfer learning y el
desglose por grupo de frecuencia del escenario long-tail.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

#: Nombres legibles para las columnas de la tabla final.
PRETTY = {
    "exp_id": "ID", "name": "Modelo", "group": "Bloque", "seed": "Semilla",
    "top1": "Top-1", "top5": "Top-5", "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted-F1", "params_total_m": "Params (M)",
    "params_trainable_m": "Entrenables (M)", "gpu_hours": "GPU-h",
    "epochs_run": "Epocas", "best_epoch": "Mejor ep.", "peak_memory_gb": "Mem (GB)",
    "throughput_img_s": "img/s", "optimizer": "Optimizador", "lr": "LR",
    "arch": "Arquitectura", "image_size": "Resolucion", "stopped_early": "Early stop",
    "augmentation": "Augmentation", "pretrained": "Preentrenado",
    "freeze_mode": "Congelamiento", "scenario": "Escenario",
    "train_val_gap": "Brecha train-val",
}


#: Esquema de la tabla de resultados. Se declara explicitamente para que un
#: directorio sin corridas devuelva un DataFrame vacio *con columnas*: asi el
#: notebook se puede ejecutar de principio a fin antes de que existan
#: resultados, y los filtros por columna no fallan.
COLUMNS = [
    "exp_id", "name", "group", "seed", "run_name", "arch", "pretrained",
    "freeze_mode", "unfrozen_blocks", "image_size", "scenario", "batch_size",
    "optimizer", "lr", "weight_decay", "dropout", "loss", "class_weighting",
    "weighted_sampler", "augmentation", "epochs_planned", "top1", "top5",
    "macro_f1", "weighted_f1", "val_loss", "params_total_m", "params_trainable_m",
    "gpu_hours", "epochs_run", "best_epoch", "stopped_early", "peak_memory_gb",
    "throughput_img_s", "train_val_gap", "by_group", "by_taxon", "notes",
]


def empty_results() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS})


def load_results(results_dir: Path) -> pd.DataFrame:
    """Lee todos los JSON de corridas completadas en un DataFrame plano."""
    rows = []
    for path in sorted(Path(results_dir).glob("*.json")):
        if path.name.startswith("status_") or path.name.endswith(".failed.json"):
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        cfg = blob["config"]
        final = blob.get("final", {})
        history = blob.get("history", [])
        rows.append(
            {
                "exp_id": blob["exp_id"], "name": blob["name"], "group": blob["group"],
                "seed": blob["seed"], "run_name": path.stem,
                "arch": cfg["model"]["arch"],
                "pretrained": cfg["model"]["pretrained"],
                "freeze_mode": cfg["model"]["freeze_mode"],
                "unfrozen_blocks": cfg["model"]["partial_unfrozen_blocks"],
                "image_size": cfg["data"]["image_size"],
                "scenario": cfg["data"]["scenario"],
                "batch_size": cfg["data"]["batch_size"],
                "optimizer": cfg["optim"]["optimizer"],
                "lr": cfg["optim"]["lr"],
                "weight_decay": cfg["optim"]["weight_decay"],
                "dropout": cfg["model"]["dropout"],
                "loss": cfg["optim"]["loss"],
                "class_weighting": cfg["optim"]["class_weighting"],
                "weighted_sampler": cfg["optim"]["weighted_sampler"],
                "augmentation": _augmentation_label(cfg["data"]),
                "epochs_planned": cfg["optim"]["epochs"],
                "top1": final.get("top1"), "top5": final.get("top5"),
                "macro_f1": final.get("macro_f1"), "weighted_f1": final.get("weighted_f1"),
                "val_loss": final.get("loss"),
                "params_total_m": blob["params_total"] / 1e6,
                "params_trainable_m": blob["params_trainable"] / 1e6,
                "gpu_hours": blob["gpu_hours"], "epochs_run": blob["epochs_run"],
                "best_epoch": blob["best_epoch"] + 1,
                "stopped_early": blob["stopped_early"],
                "peak_memory_gb": blob["peak_memory_gb"],
                "throughput_img_s": blob["throughput_img_s"],
                # Diferencia entre entrenamiento y validacion en la mejor epoca:
                # lectura directa del sobreajuste.
                "train_val_gap": _train_val_gap(history, blob["best_epoch"]),
                "by_group": blob.get("by_group", {}),
                "by_taxon": blob.get("by_taxon", {}),
                "notes": cfg.get("notes", ""),
            }
        )
    if not rows:
        return empty_results()
    return pd.DataFrame(rows).sort_values(["group", "exp_id", "seed"]).reset_index(drop=True)


def _augmentation_label(data_cfg: dict) -> str:
    """Resume la augmentation activa en una etiqueta corta."""
    parts = []
    if data_cfg["random_resized_crop"]:
        parts.append("rrc")
    if data_cfg["horizontal_flip"]:
        parts.append("flip")
    if data_cfg["color_jitter"]:
        parts.append(f"jitter{data_cfg['color_jitter']:g}")
    if data_cfg["rotation_degrees"]:
        parts.append(f"rot{data_cfg['rotation_degrees']:g}")
    if data_cfg["random_erasing"]:
        parts.append(f"erase{data_cfg['random_erasing']:g}")
    return "+".join(parts) if parts else "ninguna"


def _train_val_gap(history: list, best_epoch: int) -> float | None:
    if not history:
        return None
    idx = best_epoch if 0 <= best_epoch < len(history) else len(history) - 1
    return history[idx]["train_top1"] - history[idx]["val_top1"]


def history_frame(results_dir: Path, run_names: list[str] | None = None) -> pd.DataFrame:
    """Curvas de aprendizaje de todas las corridas, en formato largo."""
    rows = []
    for path in sorted(Path(results_dir).glob("*.json")):
        if path.name.startswith("status_") or path.name.endswith(".failed.json"):
            continue
        if run_names is not None and path.stem not in run_names:
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        for record in blob.get("history", []):
            rows.append({"run_name": path.stem, "exp_id": blob["exp_id"],
                         "name": blob["name"], "group": blob["group"],
                         "seed": blob["seed"], **record})
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in
                             ["run_name", "exp_id", "name", "group", "seed", "epoch",
                              "lr", "train_loss", "train_top1", "val_loss", "val_top1",
                              "val_top5", "val_macro_f1", "val_weighted_f1",
                              "train_seconds", "val_seconds", "peak_memory_gb"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Vistas especificas
# ---------------------------------------------------------------------------


def final_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Tabla comparativa final, en el formato que pide el enunciado."""
    cols = ["exp_id", "name", "group", "arch", "image_size", "optimizer", "lr",
            "augmentation", "pretrained", "freeze_mode", "scenario",
            "top1", "top5", "macro_f1", "weighted_f1",
            "params_trainable_m", "gpu_hours", "epochs_run"]
    out = frame[[c for c in cols if c in frame.columns]].copy()
    for col in ("top1", "top5", "macro_f1", "weighted_f1"):
        if col in out:
            out[col] = (out[col] * 100).round(2)
    for col in ("params_trainable_m", "gpu_hours"):
        if col in out:
            out[col] = out[col].round(2)
    return out.rename(columns=PRETTY)


def aggregate_seeds(frame: pd.DataFrame, keys=("exp_id", "name", "group")) -> pd.DataFrame:
    """Promedia las corridas que solo difieren en la semilla y reporta la dispersion."""
    metrics = ["top1", "top5", "macro_f1", "weighted_f1", "gpu_hours",
               "params_trainable_m", "peak_memory_gb", "train_val_gap"]
    present = [m for m in metrics if m in frame.columns]
    grouped = frame.groupby(list(keys), as_index=False).agg(
        n_seeds=("seed", "nunique"),
        **{f"{m}_mean": (m, "mean") for m in present},
        **{f"{m}_std": (m, "std") for m in present},
    )
    return grouped.sort_values(list(keys)).reset_index(drop=True)


def transfer_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Resumen del barrido de profundidad de descongelamiento.

    Anade las dos columnas que convierten el barrido en una respuesta a la
    pregunta de investigacion: la ganancia sobre el backbone congelado y la
    fraccion de la ganancia total del fine-tuning completo que cada condicion
    recupera. La segunda es la lectura directa de "cuanto rendimiento compra
    cada hora de GPU adicional".
    """
    tl = frame[frame["group"] == "transfer"].copy()
    if tl.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in
                             ["k", "exp_id", "name", "pretrained", "n", "top1",
                              "top1_std", "top5", "macro_f1", "macro_f1_std",
                              "gpu_hours", "params_trainable_m", "peak_memory_gb",
                              "throughput_img_s", "epochs_run"]})
    tl["k"] = np.where(tl["freeze_mode"] == "full", 0,
                       np.where(tl["freeze_mode"] == "none", 4, tl["unfrozen_blocks"]))
    tl.loc[~tl["pretrained"], "k"] = -1  # el control desde cero

    agg = tl.groupby(["k", "exp_id", "name", "pretrained"], as_index=False).agg(
        n=("seed", "nunique"),
        top1=("top1", "mean"), top1_std=("top1", "std"),
        top5=("top5", "mean"), macro_f1=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        gpu_hours=("gpu_hours", "mean"),
        params_trainable_m=("params_trainable_m", "mean"),
        peak_memory_gb=("peak_memory_gb", "mean"),
        throughput_img_s=("throughput_img_s", "mean"),
        epochs_run=("epochs_run", "mean"),
    ).sort_values("k")

    pre = agg[agg["pretrained"]]
    if not pre.empty:
        frozen = pre[pre["k"] == 0]["top1"]
        full = pre[pre["k"] == 4]["top1"]
        if len(frozen) and len(full):
            base, top = float(frozen.iloc[0]), float(full.iloc[0])
            agg["ganancia_vs_congelado"] = agg["top1"] - base
            span = top - base
            agg["frac_ganancia_full"] = (
                (agg["top1"] - base) / span if abs(span) > 1e-9 else np.nan
            )
            # Puntos de Top-1 sobre el congelado por hora de GPU adicional.
            extra_h = agg["gpu_hours"] - float(pre[pre["k"] == 0]["gpu_hours"].iloc[0])
            agg["top1_por_gpu_hora_extra"] = np.where(
                extra_h > 1e-6, agg["ganancia_vs_congelado"] / extra_h, np.nan
            )
    return agg.reset_index(drop=True)


def longtail_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Rendimiento por grupo de frecuencia para cada estrategia del bloque long-tail."""
    rows = []
    for _, run in frame[frame["group"] == "longtail"].iterrows():
        row = {"exp_id": run["exp_id"], "name": run["name"],
               "scenario": run["scenario"], "loss": run["loss"],
               "class_weighting": run["class_weighting"],
               "weighted_sampler": run["weighted_sampler"],
               "top1": run["top1"], "macro_f1": run["macro_f1"]}
        for group in ("head", "mid", "tail"):
            stats = (run["by_group"] or {}).get(group, {})
            row[f"{group}_recall"] = stats.get("recall")
            row[f"{group}_f1"] = stats.get("macro_f1")
            row[f"{group}_n_clases"] = stats.get("n_classes")
        # Brecha cabeza-cola: la medida directa de cuanto dano hace el desbalance.
        if row.get("head_recall") is not None and row.get("tail_recall") is not None:
            row["brecha_head_tail"] = row["head_recall"] - row["tail_recall"]
        rows.append(row)
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in
                             ["exp_id", "name", "scenario", "loss", "class_weighting",
                              "weighted_sampler", "top1", "macro_f1", "head_recall",
                              "mid_recall", "tail_recall", "head_f1", "mid_f1",
                              "tail_f1", "brecha_head_tail"]})
    return pd.DataFrame(rows).sort_values("exp_id").reset_index(drop=True)


def taxon_summary(frame: pd.DataFrame, run_name: str) -> pd.DataFrame:
    """Rendimiento por grupo taxonomico de una corrida concreta."""
    match = frame[frame["run_name"] == run_name]
    if match.empty:
        return pd.DataFrame()
    by_taxon = match.iloc[0]["by_taxon"] or {}
    return (
        pd.DataFrame([{"taxon": k, **v} for k, v in by_taxon.items()])
        .sort_values("macro_f1", ascending=False)
        .reset_index(drop=True)
    )


def ablation_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Escalera de ablation en el formato del enunciado."""
    order = ["R1", "R2", "R3", "R4", "R5", "R6"]
    sub = frame[frame["exp_id"].isin(order)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["Exp.", "Configuracion", "Augmentation",
                                     "Dropout", "Weight decay", "Top-1", "Macro-F1",
                                     "Brecha train-val", "Delta vs anterior"])
    sub["Augmentation"] = sub["augmentation"]
    sub["Dropout"] = np.where(sub["dropout"] > 0, f"si", "no")
    sub["Weight decay"] = np.where(sub["weight_decay"] > 0, "si", "no")
    sub["Top-1"] = (sub["top1"] * 100).round(2)
    sub["Macro-F1"] = (sub["macro_f1"] * 100).round(2)
    sub["Brecha train-val"] = (sub["train_val_gap"] * 100).round(2)
    sub["Delta vs anterior"] = sub["Top-1"].diff().round(2)
    cols = ["exp_id", "name", "Augmentation", "Dropout", "Weight decay",
            "Top-1", "Macro-F1", "Brecha train-val", "Delta vs anterior"]
    return sub[cols].rename(columns={"exp_id": "Exp.", "name": "Configuracion"}).reset_index(drop=True)
