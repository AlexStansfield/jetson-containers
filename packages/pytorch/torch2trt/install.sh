#!/usr/bin/env bash
set -ex

cd /opt
git clone --depth=1 https://github.com/NVIDIA-AI-IOT/torch2trt

cd torch2trt
ls -R /tmp/torch2trt
cp /tmp/torch2trt/flattener.py torch2trt

# Patch: force PyTorch CUDA init before torch2trt loads
# JetPack 6.x deferred CUDA init can cause torch2trt to see stub driver (Error 34)
python3 - <<'PY'
from pathlib import Path

init_path = Path('torch2trt/__init__.py')
prefix = (
    "# [patch] Force CUDA init early to avoid stub driver error on Jetson\n"
    "import torch as _torch\n"
    "try:\n"
    "    _ = _torch.cuda.is_available()  # triggers CUDA runtime load\n"
    "except Exception:\n"
    "    pass\n\n"
)

if init_path.exists():
    original = init_path.read_text()
    # only prepend if not already patched
    if "Force CUDA init early" not in original:
        init_path.write_text(prefix + original)
        print(f"Prepended CUDA init to {init_path}")
    else:
        print(f"CUDA init patch already present in {init_path}")
else:
    # create __init__.py if missing
    init_path.write_text(prefix)
    print(f"Created {init_path} with CUDA init patch")
PY
# END Patch

python3 setup.py install --plugins

sed 's|^set(CUDA_ARCHITECTURES.*|#|g' -i CMakeLists.txt
sed 's|Catch2_FOUND|False|g' -i CMakeLists.txt

cmake -B build \
  -DCUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 .

cmake --build build --target install

ldconfig

uv pip install nvidia-pyindex
uv pip install onnx-graphsurgeon
