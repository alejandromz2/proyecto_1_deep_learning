#!/bin/bash
# Envia solo segmentos de GPU, para cuando el dataset ya esta preparado y
# verificado. Es el script que usa un colaborador que reutiliza un corpus ya
# construido, o quien retoma una cola interrumpida.
#
#   bash khipu/submit_gpu.sh [numero_de_segmentos]
#
# Los segmentos se encadenan con `afterany` y no con `afterok`: cada uno retoma
# la cola donde quedo el anterior, de modo que un fallo aislado no cancele el
# resto. El runner salta las corridas que ya tienen su JSON, asi que sobrar
# segmentos no cuesta nada: el ultimo termina en segundos si no queda trabajo.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results

SEGMENTOS="${1:-3}"
IDS="khipu/job_ids.tsv"

echo "Pendientes antes de enviar:"
module load python3/3.13.2
PYTHONPATH="$PWD/src" INAT_DATA_ROOT="${INAT_DATA_ROOT:-$HOME/datasets/inat2021}" \
  .venv/bin/python -m inat.runner --phase all --dry-run --results results 2>/dev/null \
  | grep -c '\[pend' || true

prev=""
for seg in $(seq 1 "$SEGMENTOS"); do
  if [[ -z "$prev" ]]; then
    prev=$(sbatch --parsable --job-name="inat-g${seg}" khipu/job_gpu.sh all 7.5)
  else
    prev=$(sbatch --parsable --job-name="inat-g${seg}" \
           --dependency="afterany:${prev}" khipu/job_gpu.sh all 7.5)
  fi
  printf 'gpu-%s\t%s\n' "$seg" "$prev" >> "$IDS"
  echo "  segmento $seg -> $prev"
done

squeue -u "$USER" -o '%.8i %.12j %.9T %.9M %.20R'
