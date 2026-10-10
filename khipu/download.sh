#!/bin/bash
# Descarga del corpus desde S3. Se ejecuta en el NODO DE ACCESO porque los
# nodos de computo no tienen salida a internet. Es trabajo de entrada/salida,
# no de calculo: ocho conexiones de curl sin uso apreciable de CPU.
#
#   nohup bash khipu/download.sh > logs/download.log 2>&1 &
#
set -eo pipefail
cd "$(dirname "$0")/.."
module load python3/3.13.2
set -u
export INAT_DATA_ROOT="${INAT_DATA_ROOT:-$HOME/datasets/inat2021}"
mkdir -p logs "$INAT_DATA_ROOT"
.venv/bin/python scripts/prepare_data.py download --data-root "$INAT_DATA_ROOT" --connections 8
echo "descarga completa en $INAT_DATA_ROOT/raw"
