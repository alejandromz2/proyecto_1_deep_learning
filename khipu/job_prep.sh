#!/bin/bash
# Preparacion del dataset: descomprimir, redimensionar a lado corto 256 px y
# construir el indice Parquet. Es trabajo intensivo de CPU y no necesita red,
# asi que va a la particion `standard` y deja libre la GPU.
#
# La fase por omision es `prepare` y NO incluye la descarga: los nodos de
# computo de Khipu no resuelven DNS, de modo que un `all` aqui abortaria el
# job y, por la dependencia afterok, toda la cadena.
#SBATCH --job-name=inat-prep
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.log
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load python3/3.13.2
set -u

export INAT_DATA_ROOT="${INAT_DATA_ROOT:-$HOME/datasets/inat2021}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

echo "nodo: $(hostname)   cpus: ${SLURM_CPUS_PER_TASK}"
srun .venv/bin/python scripts/prepare_data.py "${1:-prepare}" \
    --data-root "$INAT_DATA_ROOT" \
    --workers "${SLURM_CPUS_PER_TASK}"
