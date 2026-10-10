# Instrucciones para ejecutar las corridas pendientes en Khipu

Estas instrucciones son para **ejecutar en tu propia cuenta de Khipu** las 16 corridas que
faltan del Proyecto #1, y devolver los resultados al repositorio común.

No necesitas conocer el proyecto por dentro. Todo el diseño experimental está declarado en
código: tú lanzas el ejecutor y él decide qué falta.

---

## Qué vas a correr y por qué tú

El proyecto son 54 corridas de entrenamiento sobre iNaturalist 2021 Mini (500,000 imágenes,
10,000 especies). **38 ya están hechas.** Faltan 16, que son el bloque principal:

| Bloque | Corridas | Tiempo aproximado de GPU |
|---|---:|---:|
| Sonda de learning rate que quedó pendiente | 1 | 0.2 h |
| **Transfer Learning** (barrido k = 0…4 + control) | 9 | ~10 h |
| **Long-Tail Challenge** | 6 | ~3 h |
| | **16** | **~13 h** |

En la cuenta original la A100 completa está ocupada por otro usuario hasta el 10 de octubre.
Lanzarlo desde una segunda cuenta da una segunda posición en la cola y, como tu cuenta no ha
consumido GPU recientemente, es probable que Slurm te asigne mayor prioridad.

**No hay riesgo de duplicar trabajo**: el ejecutor salta cualquier corrida que ya tenga su
archivo de resultados, y el repositorio los trae consigo.

---

## Antes de empezar

Necesitas:

- Cuenta en Khipu (`ssh <usuario>@khipu.utec.edu.pe`)
- Acceso al repositorio privado `pjcaycho/proyecto_1_deep_learning` (pídelo y acepta la
  invitación)
- Unos 25 GB libres en tu `$HOME` si vas a preparar tu propia copia del dataset

Comprueba tu cuota de Slurm:

```bash
sacctmgr show assoc where user=$USER format=User,Account,QOS
```

Deberías ver la cuenta `postgrado` con QOS `a-postgrado`: 8 horas por job y una GPU a la vez.

---

## Paso 1 — Clonar el repositorio

```bash
mkdir -p ~/experiments && cd ~/experiments
git clone https://github.com/pjcaycho/proyecto_1_deep_learning.git inat2021_proyecto1
cd inat2021_proyecto1
```

Si pide contraseña, usa un **Personal Access Token** de GitHub, no tu contraseña
(github.com → Settings → Developer settings → Personal access tokens → Fine-grained, con
permiso de lectura/escritura sobre ese repositorio).

El repositorio ya trae en `results/` los 38 resultados existentes. **Eso es lo que hace que
el ejecutor sepa qué no repetir.**

---

## Paso 2 — Crear el entorno

```bash
bash khipu/setup.sh
```

Tarda unos 10 minutos. Crea `.venv`, instala PyTorch 2.11 con CUDA 12.8 y **descarga los
pesos de ImageNet a `.torch/`**.

> Esto debe correrse en el **nodo de acceso**, no dentro de un job. Los nodos de cómputo de
> Khipu no resuelven DNS: si los pesos no quedan en caché ahora, los experimentos de
> transfer learning fallarían ya en la cola.

Verifica:

```bash
module load python3/3.13.2
.venv/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
ls .torch/hub/checkpoints/        # deben aparecer 6 archivos .pth
```

(`cuda.is_available()` dirá `False` en el nodo de acceso, que no tiene GPU. Es lo esperado.)

---

## Paso 3 — Conseguir el dataset

Hay dos caminos. **El A es mucho más rápido**, pero requiere que el dueño de la cuenta
original abra permisos.

### Opción A — Reutilizar el corpus ya preparado (minutos)

Pídele al dueño de la cuenta original que ejecute, una sola vez:

```bash
chmod o+x /home/pedro.caycho
chmod -R o+rX /home/pedro.caycho/datasets/inat2021
```

Luego, en tu cuenta, comprueba que lo ves y úsalo directamente:

```bash
export INAT_DATA_ROOT=/home/pedro.caycho/datasets/inat2021
ls $INAT_DATA_ROOT/index/        # train.parquet, val.parquet, categories.parquet
```

Solo necesitas lectura: el entrenamiento nunca escribe en el dataset.

### Opción B — Preparar tu propia copia (~6 horas, desatendido)

```bash
export INAT_DATA_ROOT=$HOME/datasets/inat2021
mkdir -p logs $INAT_DATA_ROOT

# Descarga (~50 GB desde S3). En el nodo de acceso: necesita red.
nohup bash khipu/download.sh > logs/download.log 2>&1 &

# Cuando termine (unos 35 min), preparación en un nodo de CPU (~5 h)
sbatch khipu/job_prep.sh prepare
```

La preparación descomprime, redimensiona las 600,000 imágenes a lado corto 256 px (de 50 GB
a 18 GB, que es lo que hace viable entrenar sobre el corpus completo) y construye los
índices. Al final verifica que cada anotación tenga su imagen.

El resultado es **idéntico** al de la cuenta original: el redimensionado es determinista, así
que los resultados siguen siendo comparables.

---

## Paso 4 — Comprobar antes de lanzar

```bash
export INAT_DATA_ROOT=<la ruta que elegiste>
module load python3/3.13.2
PYTHONPATH=$PWD/src .venv/bin/python khipu/preflight.py
```

Debe decir `PREFLIGHT OK`. Comprueba sintaxis, pesos en caché, índices y que las imágenes
estén donde el índice dice.

Y mira qué va a correr:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m inat.runner \
    --phase all --dry-run --data-root "$INAT_DATA_ROOT" --results results
```

Debe listar **16 pendientes** y el resto como `[hecho]`. Si lista 54, el `results/` del
repositorio no se clonó bien.

---

## Paso 5 — Lanzar

```bash
bash khipu/submit_gpu.sh 3
```

Envía 3 segmentos de GPU de hasta 8 horas, encadenados. Cada uno procesa la cola durante
7.5 h y cierra ordenadamente; el siguiente retoma donde quedó. Sobrar segmentos no cuesta
nada: si no queda trabajo, el último termina en segundos.

Están encadenados con `afterany` y no con `afterok` **a propósito**: un fallo aislado no debe
cancelar los que vienen detrás.

Para seguirlo:

```bash
bash khipu/status.sh                      # cola, historial y avance
squeue -u $USER                           # solo la cola
tail -f logs/inat-g1-*.log                # el entrenamiento en vivo
ls results/*.json | grep -vc status_      # cuántas van
```

**No hace falta mantener la sesión abierta.** Los jobs siguen bajo Slurm aunque cierres el
terminal o apagues tu computadora.

Una línea de entrenamiento se ve así:

```
  ep  7/15  train 2.431/0.412  val 2.198/0.455 top5 0.712 mF1 0.431  312s
```

época, pérdida y Top-1 de entrenamiento, lo mismo en validación, Top-5, Macro-F1 y segundos.

---

## Paso 6 — Devolver los resultados

Cada corrida deja un JSON en `results/` con su configuración completa y su curva de
aprendizaje. Además, las corridas del bloque principal guardan los pesos del mejor modelo en
`results/models/` (unos 170 MB cada una).

```bash
cd ~/experiments/inat2021_proyecto1
git config user.name  "Tu Nombre"
git config user.email "tu@correo"

git add results/*.json
git commit -m "Resultados de transfer learning y long-tail ejecutados en <tu usuario>"
git push origin main
```

> **No subas `results/models/*.pt` al repositorio**: son varios GB y `.gitignore` ya los
> excluye. Esos pesos se necesitan para el análisis de errores; coordina con el equipo si hay
> que moverlos (lo normal es dejarlos en Khipu y correr ahí ese análisis).

Si `git push` rechaza por cambios en el remoto:

```bash
git pull --rebase origin main
git push origin main
```

No habrá conflictos de contenido: cada corrida escribe su propio archivo con nombre único.

---

## Consolidación final

Una vez estén los 54 resultados en el repositorio, el notebook se regenera **en menos de diez
minutos**, porque no entrena nada: lee los JSON y dibuja.

```bash
cd ~/experiments/inat2021_proyecto1
export INAT_DATA_ROOT=<ruta al dataset>
module load python3/3.13.2
PYTHONPATH=$PWD/src MPLBACKEND=Agg .venv/bin/python scripts/build_notebook.py
PYTHONPATH=$PWD/src MPLBACKEND=Agg .venv/bin/python - <<'PY'
import nbformat
from nbclient import NotebookClient
p = "notebooks/proyecto1_biodiversity_at_scale.ipynb"
nb = nbformat.read(p, as_version=4)
NotebookClient(nb, timeout=3600, kernel_name="python3",
               resources={"metadata": {"path": "notebooks/"}}).execute()
nbformat.write(nb, p)
print("notebook actualizado")
PY
```

El notebook trae una sección de avance que se recalcula sola: compara el diseño declarado en
`inat.registry` contra los archivos en disco y dice qué bloques están completos.

La única parte que necesita GPU es el análisis de errores, que carga el mejor modelo y corre
inferencia sobre las 100,000 imágenes de validación. Si lo vas a ejecutar, hazlo dentro de un
job, no en el nodo de acceso.

---

## Si algo sale mal

| Síntoma | Causa y solución |
|---|---|
| `PREFLIGHT FALLIDO: falta .../train.parquet` | El dataset no está preparado o `INAT_DATA_ROOT` apunta mal. Paso 3. |
| `no hay pesos preentrenados en cache` | `khipu/setup.sh` no se completó. Vuelve a correrlo **en el nodo de acceso**. |
| El dry-run lista 54 pendientes | `results/` llegó vacío. Comprueba que el clon trajo los JSON. |
| `Too many open files` | Ya está corregido en el repositorio. Asegúrate de haber clonado la versión actual (`git pull`). |
| Job en `PENDING (Resources)` mucho tiempo | La A100 está ocupada. `squeue --start -j <id>` estima cuándo arranca. |
| Job `CANCELLED` sin haber empezado | Dependencia incumplida. Con `submit_gpu.sh` no debería pasar: usa `afterany`. |
| `libpython3.13.so.1.0: cannot open shared object file` | Falta `module load python3/3.13.2` antes de usar `.venv/bin/python`. |

Si una corrida falla, deja un `results/<nombre>.failed.json` con el traceback y **la cola
continúa**. Para reintentarla, borra ese archivo y vuelve a lanzar.

---

## Resumen

```bash
git clone https://github.com/pjcaycho/proyecto_1_deep_learning.git inat2021_proyecto1
cd inat2021_proyecto1
bash khipu/setup.sh
export INAT_DATA_ROOT=/home/pedro.caycho/datasets/inat2021   # u opción B
module load python3/3.13.2
PYTHONPATH=$PWD/src .venv/bin/python khipu/preflight.py
bash khipu/submit_gpu.sh 3
# ...esperar...
git add results/*.json && git commit -m "resultados" && git push origin main
```
