"""Descarga los resultados desde Khipu sin modificar nada en el clúster.

    KHIPU_CONNECTION=<ruta> KHIPU_USER=<usuario> python khipu/sync_results.py [--models] [--logs]

Por omision trae solo los JSON de las corridas y la tabla final, que es lo que
el notebook necesita. Los pesos (`--models`) son ~100 MB por corrida y solo
hacen falta para el analisis de errores.
"""
from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path

# La conexion SSH (credenciales incluidas) vive FUERA del repositorio a
# proposito. Se indica su ruta con KHIPU_CONNECTION; debe contener un
# `remote.py` que exponga `connect()`.
CONNECTION = Path(os.environ.get("KHIPU_CONNECTION", "../khipu_connection")).expanduser()
sys.path.insert(0, str(CONNECTION))

REMOTE = os.environ.get("KHIPU_REMOTE", "~/experiments/inat2021_proyecto1").replace(
    "~", f"/home/{os.environ.get('KHIPU_USER', 'USUARIO')}"
)
LOCAL = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", action="store_true", help="incluir pesos (~100 MB/corrida)")
    parser.add_argument("--logs", action="store_true", help="incluir logs de Slurm")
    args = parser.parse_args()

    from remote import connect

    client = connect()
    try:
        with client.open_sftp() as sftp:

            def download(remote: str, local: Path, skip_models: bool) -> int:
                n = 0
                local.mkdir(parents=True, exist_ok=True)
                for entry in sftp.listdir_attr(remote):
                    source, target = f"{remote}/{entry.filename}", local / entry.filename
                    if stat.S_ISDIR(entry.st_mode):
                        if skip_models and entry.filename in ("models", "checkpoints"):
                            continue
                        n += download(source, target, skip_models)
                    elif (not target.exists()
                          or target.stat().st_size != entry.st_size
                          or target.stat().st_mtime < entry.st_mtime):
                        # Descarga a un temporal y renombra: una interrupcion no
                        # deja un JSON a medias que el analisis leeria como valido.
                        tmp = target.with_name(target.name + ".part")
                        sftp.get(source, str(tmp))
                        tmp.replace(target)
                        n += 1
                return n

            total = download(f"{REMOTE}/results", LOCAL / "results", skip_models=not args.models)
            print(f"results: {total} archivos nuevos o actualizados")
            if args.logs:
                total = download(f"{REMOTE}/logs", LOCAL / "logs", skip_models=True)
                print(f"logs: {total} archivos")
            for name in ("khipu/environment.lock.txt", "khipu/job_ids.tsv"):
                try:
                    (LOCAL / name).parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(f"{REMOTE}/{name}", str(LOCAL / name))
                except FileNotFoundError:
                    pass
    finally:
        client.close()
    print(f"descargado en {LOCAL / 'results'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
