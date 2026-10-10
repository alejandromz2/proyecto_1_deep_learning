# Biodiversity at Scale — Clasificación fine-grained de especies

Proyecto #1 del curso **Aprendizaje Profundo (Práctica)**, Maestría de Investigación en
Inteligencia Artificial, UTEC. Dataset: **iNaturalist 2021 Mini** (500,000 imágenes de
entrenamiento, 100,000 de validación, 10,000 especies).

## Preguntas de investigación

> **P1.** ¿El Fine-Tuning completo supera a utilizar un backbone congelado?
>
> **P2.** ¿El Partial Fine-Tuning proporciona una mejor relación entre rendimiento y costo
> computacional?

Ambas se responden con un único experimento controlado: un **barrido sobre la profundidad
de descongelamiento** `k` (número de etapas finales del backbone que se entrenan), donde
`k=0` es feature extraction, `k=1..3` es fine-tuning parcial y `k=4` es fine-tuning
completo. Se incluye un control entrenado desde cero con el mismo presupuesto, sin el cual
no es posible separar el aporte del preentrenamiento del aporte de la arquitectura.

## Estructura del repositorio

```
.
├── notebooks/
│   └── proyecto1_biodiversity_at_scale.ipynb   # entregable principal: narrativa y análisis
├── src/inat/
│   ├── config.py      # ExperimentConfig: todo lo que distingue una corrida de otra
│   ├── seeding.py     # control centralizado de aleatoriedad
│   ├── data.py        # índice, escenarios (balanceado / long-tail), Dataset, DataLoaders
│   ├── models.py      # MLP, CNN de torchvision, estrategias de congelamiento
│   ├── engine.py      # bucle de entrenamiento, AMP, early stopping, checkpoints
│   ├── metrics.py     # Top-k, Macro/Weighted-F1, desglose por grupo y por taxón
│   ├── registry.py    # DISEÑO EXPERIMENTAL: las 54 corridas, declarativas
│   ├── runner.py      # ejecutor con presupuesto de tiempo y reanudación
│   ├── analysis.py    # agregación de resultados a tablas
│   ├── plots.py       # figuras
│   ├── eda.py         # análisis exploratorio
│   └── errors.py      # análisis de errores del mejor modelo
├── scripts/
│   ├── prepare_data.py    # descarga, extracción, redimensionado, índice
│   ├── make_dev_subset.py # índice de desarrollo a partir de imágenes locales
│   ├── smoke_test.py      # prueba de integridad del pipeline
│   └── build_notebook.py  # genera el notebook
├── khipu/                 # ejecución en el clúster (Slurm)
│   ├── setup.sh, download.sh, job_prep.sh, job_gpu.sh, submit.sh, status.sh, preflight.py
├── results/               # JSON por corrida, tabla final, pesos del mejor modelo
├── figures/               # figuras generadas
└── requirements.txt
```

---

## 1. Obtener el dataset

Las imágenes **no están en el repositorio**. Se descargan de los buckets oficiales:

```bash
python scripts/prepare_data.py download --data-root <RUTA> --connections 8
```

Descarga `train_mini.tar.gz` (~42 GB), `val.tar.gz` (~8.4 GB) y las anotaciones, usando
peticiones por rango en paralelo. Requiere salida a internet.

## 2. Preparar los datos

```bash
python scripts/prepare_data.py all --data-root <RUTA> --workers 32
```

Fases: `download` → `extract` → `resize` → `index` → `verify`.

El paso **`resize`** genera una copia con el **lado corto fijo en 256 px** (~13 GB). Es la
decisión de ingeniería que hace viable el corpus completo: entrenar sobre los JPEG
originales obliga a decodificar 600,000 imágenes de hasta 500 px por época, lo que satura
la CPU y deja la GPU esperando datos. Es la única transformación irreversible del pipeline
y se aplica antes de cualquier experimento, de modo que **todos los modelos ven exactamente
los mismos píxeles**.

El paso **`index`** convierte los JSON estilo COCO (208 MB) en tablas Parquet que el resto
del proyecto consume.

### Subset de desarrollo

Para abrir el notebook o depurar sin descargar 50 GB:

```bash
python scripts/make_dev_subset.py --source <carpeta con los JSON e imágenes> --out <RUTA_DEV>
```

Genera un índice con el mismo esquema, restringido a las imágenes presentes en disco.

## 3. Verificar el pipeline

```bash
python scripts/smoke_test.py --data-root <RUTA>
```

Ejercita cada camino distinto del código —baseline denso, CNN desde cero, las tres
estrategias de transfer learning, el escenario long-tail y cada mitigación— y comprueba que
el congelamiento deja exactamente los parámetros que debe, que las métricas son coherentes
y que reanudar desde un checkpoint reproduce el estado.

## 4. Ejecutar los experimentos

```bash
# Ver la cola sin entrenar
python -m inat.runner --phase all --dry-run --data-root <RUTA>

# Una fase concreta
python -m inat.runner --phase transfer --budget-hours 7.5 --data-root <RUTA>

# Toda la cola
python -m inat.runner --phase all --budget-hours 7.5 --data-root <RUTA>
```

Fases, en el orden en que deben ejecutarse:

| Fase | Corridas | Contenido |
|---|---:|---|
| `baseline` | 4 | MLP a tres resoluciones + ablation de BatchNorm |
| `arquitectura` | 3 | ResNet-50, EfficientNet-B0, MobileNetV3-Large desde cero |
| `optimizacion` | 11 | 5 optimizadores + 6 sondas de sensibilidad al learning rate |
| `regularizacion` | 6 | Escalera de ablation + color jitter + random erasing |
| `transfer_lr` | 15 | Barrido de learning rate por profundidad de descongelamiento |
| `transfer` | 9 | **Bloque principal**: k = 0…4 + control desde cero |
| `longtail` | 6 | 2 controles + sin corrección + 3 mitigaciones |

El runner **salta las corridas ya completadas** (existe su JSON) y guarda un checkpoint por
época, de modo que la cola puede atravesar varios jobs sin repetir trabajo.

## 5. Reproducir el análisis

```bash
python scripts/build_notebook.py          # regenera el notebook
jupyter lab notebooks/proyecto1_biodiversity_at_scale.ipynb
```

El notebook **no entrena**: carga los JSON de `results/` y produce las tablas y figuras. Es
ejecutable de principio a fin aunque todavía no existan resultados (las secciones
pendientes lo indican), lo que evita que el informe dependa de celdas ejecutadas a mano en
un orden particular.

Variables de entorno: `INAT_DATA_ROOT` (corpus preparado), `INAT_RESULTS`, `INAT_FIGURES`.

---

## Ejecución en Khipu (UTEC)

El clúster impone dos restricciones que determinan el diseño de los jobs:

- La cuenta `postgrado` (QOS `a-postgrado`) permite **8 horas por job** y **una GPU**.
- **Los nodos de cómputo no tienen salida a internet** (DNS no resuelve). Por eso la
  instalación del entorno, la descarga del corpus y la descarga de los pesos de ImageNet
  ocurren en el nodo de acceso; si los pesos no estuvieran en caché, los experimentos de
  transfer learning fallarían ya en la cola.

```bash
# En el nodo de acceso (requiere internet)
bash khipu/setup.sh                                    # venv + torch + pesos de ImageNet
nohup bash khipu/download.sh > logs/download.log 2>&1 & # corpus desde S3

# Cadena de jobs
python khipu/preflight.py     # comprobaciones de solo lectura
bash khipu/submit.sh          # prep (CPU) → verify (MIG) → N segmentos de GPU
bash khipu/status.sh          # estado
```

`submit.sh` encadena los jobs con `afterok`: un fallo detiene la cadena en lugar de
encadenar corridas sobre un estado roto. El job de preparación corre en la partición
`standard` (es trabajo de CPU y no necesita GPU); los de entrenamiento piden la A100
completa del nodo `ag001`. Cada job de GPU recibe un presupuesto de 7.5 h y Slurm le envía
`SIGUSR1` quince minutos antes del corte, lo que le da margen para guardar el checkpoint de
la época en curso y salir ordenadamente.

---

## Reproducibilidad

- **Semillas.** Una semilla por experimento alimenta inicialización, orden del `DataLoader`,
  augmentation y muestreo de clases (`inat.seeding`). Los workers reciben semillas derivadas
  y deterministas.
- **Configuración serializada.** Cada corrida escribe su `ExperimentConfig` completo junto a
  sus resultados: cada fila de la tabla final es trazable hasta sus hiperparámetros exactos.
  Un *fingerprint* SHA-256 detecta corridas incomparables.
- **Escenario long-tail construido por código.** Las especies se barajan con una semilla
  declarada y se parten en tres grupos con cuota fija. No hay selección manual de archivos
  en ninguna etapa.
- **Validación nunca usada para entrenar.** Se usa para early stopping y para elegir el
  learning rate de cada condición; ese procedimiento se aplica por igual a todas las
  condiciones comparadas.
- **Determinismo.** `torch.use_deterministic_algorithms` está disponible en `seed_everything`
  pero no se activa en el entrenamiento masivo por su costo en convoluciones: en su lugar se
  reportan varias semillas en las condiciones sobre las que se apoyan las conclusiones.

## Entorno

Python 3.13, PyTorch 2.11.0 + CUDA 12.8, torchvision 0.26.0. Ver `requirements.txt` y
`khipu/environment.lock.txt` (congelado tras el `setup.sh` del clúster).

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
