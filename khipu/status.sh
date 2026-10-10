#!/bin/bash
# Estado de la cadena de jobs de este experimento y avance de la cola.
set -euo pipefail
cd "$(dirname "$0")/.."
ids=$(cut -f2 khipu/job_ids.tsv | paste -sd,)
echo "=== cola de Slurm ==="
squeue -j "$ids" -o '%.10i %.16j %.9T %.11M %.10L %.30R' 2>/dev/null || true
echo
echo "=== historial ==="
sacct -j "$ids" --format=JobID%12,JobName%16,State%12,Elapsed,ExitCode,AllocTRES%48 | grep -v '\.batch\|\.extern' || true
echo
echo "=== avance de experimentos ==="
done=$(ls results/*.json 2>/dev/null | grep -vc 'status_\|failed' || echo 0)
failed=$(ls results/*.failed.json 2>/dev/null | wc -l || echo 0)
echo "completados: ${done}   fallidos: ${failed}"
for f in results/status_*.json; do [[ -e "$f" ]] && echo "--- $f" && cat "$f"; done
echo
echo "=== ultimo log ==="
latest=$(ls -t logs/*.log 2>/dev/null | head -1)
[[ -n "${latest:-}" ]] && tail -n 20 "$latest"
