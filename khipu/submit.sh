#!/bin/bash
# Envia la cadena completa de jobs a Slurm.
#
# Estructura:
#   prep    (CPU, particion standard)  descomprime, redimensiona e indexa
#   verify  (GPU MIG de 5 GB, 30 min)  prueba de integridad del pipeline
#   gpu1..N (A100 completa, 8 h c/u)   recorren la cola de experimentos
#
# Cada job depende del anterior con `afterok`, de modo que un fallo detiene la
# cadena en lugar de encadenar corridas sobre un estado roto. Los jobs de GPU
# comparten la misma cola: cada uno salta lo ya terminado y retoma lo pendiente,
# asi que el numero de segmentos es un presupuesto, no un plan rigido.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results

SEGMENTS="${SEGMENTS:-5}"
IDS=khipu/job_ids.tsv

if [[ -s "$IDS" ]]; then
  echo "Ya existe un envio en $IDS. Revisar con 'bash khipu/status.sh' antes de duplicar." >&2
  exit 1
fi

prev=$(sbatch --parsable khipu/job_prep.sh prepare)
printf 'prep\t%s\n' "$prev" >> "$IDS"

prev=$(sbatch --parsable --job-name=inat-verify \
       --gres=gpu:a100_1g.5gb:1 --time=00:30:00 \
       --dependency="afterok:${prev}" --kill-on-invalid-dep=yes \
       khipu/job_gpu.sh verify 0.4)
printf 'verify\t%s\n' "$prev" >> "$IDS"

for segment in $(seq 1 "$SEGMENTS"); do
  prev=$(sbatch --parsable --job-name="inat-gpu-${segment}" \
         --dependency="afterok:${prev}" --kill-on-invalid-dep=yes \
         khipu/job_gpu.sh all 7.5)
  printf 'gpu-%s\t%s\n' "$segment" "$prev" >> "$IDS"
done

cat "$IDS"
