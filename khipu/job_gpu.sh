#!/bin/bash
# Un segmento de experimentos en la A100 del nodo ag001.
#
# La cuenta `postgrado` limita cada job a 8 horas y una GPU. El runner recibe
# un presupuesto de 7.5 h y cierra ordenadamente antes del limite; la senal
# USR1 quince minutos antes del corte le da margen para guardar el checkpoint
# de la epoca en curso. Lo que no alcanzo a correr queda en la cola y lo retoma
# el siguiente job de la cadena.
#SBATCH --job-name=inat-gpu
#SBATCH --partition=gpu
#SBATCH --nodelist=ag001
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=72G
#SBATCH --time=08:00:00
#SBATCH --signal=USR1@900
#SBATCH --output=logs/%x-%j.log
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load python3/3.13.2
set -u

export INAT_DATA_ROOT="${INAT_DATA_ROOT:-$HOME/datasets/inat2021}"
export INAT_RESULTS="${INAT_RESULTS:-$PWD/results}"
export TORCH_HOME="$PWD/.torch"      # pesos de ImageNet ya descargados
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
# Los workers del DataLoader son los que decodifican JPEG; limitar los hilos de
# BLAS evita que compitan entre si por los 16 nucleos asignados.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

nvidia-smi
srun .venv/bin/python -m inat.runner \
    --phase "${1:?falta la fase}" \
    --budget-hours "${2:-7.5}" \
    --data-root "$INAT_DATA_ROOT" \
    --results "$INAT_RESULTS"
