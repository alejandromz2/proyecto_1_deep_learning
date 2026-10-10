#!/bin/bash
# Entorno del proyecto en Khipu. Se ejecuta UNA vez en el nodo de acceso,
# porque los nodos de computo no resuelven DNS y por tanto no pueden instalar
# paquetes ni descargar los pesos de ImageNet.
set -eo pipefail
cd "$(dirname "$0")/.."
module load python3/3.13.2
set -u

PY=/opt/ohpc/pub/libs/gnu12/python3/3.13.2/bin/python3
$PY -m venv .venv
.venv/bin/python -m pip install --upgrade pip wheel

# torch/torchvision con CUDA 12.8: es la combinacion verificada contra la A100
# del nodo ag001 y la misma que se usa en el entorno local de desarrollo.
.venv/bin/python -m pip install \
  torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install \
  numpy pandas pyarrow pillow matplotlib scikit-learn nbformat ipython

# Cache de pesos preentrenados dentro del proyecto. Los jobs de GPU no tienen
# salida a internet, asi que los pesos de ImageNet deben quedar en disco ahora;
# de lo contrario los experimentos de transfer learning fallarian en la cola.
export TORCH_HOME="$PWD/.torch"
mkdir -p "$TORCH_HOME"
.venv/bin/python - <<'PY'
import torchvision.models as tvm
for name, weights in [
    ("resnet50", tvm.ResNet50_Weights.IMAGENET1K_V2),
    ("resnet18", tvm.ResNet18_Weights.IMAGENET1K_V1),
    ("efficientnet_b0", tvm.EfficientNet_B0_Weights.IMAGENET1K_V1),
    ("mobilenet_v3_large", tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    ("densenet121", tvm.DenseNet121_Weights.IMAGENET1K_V1),
    ("convnext_tiny", tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
]:
    weights.get_state_dict(progress=False)
    print("pesos en cache:", name)
PY

.venv/bin/python -m pip freeze > khipu/environment.lock.txt
echo "setup completo"
