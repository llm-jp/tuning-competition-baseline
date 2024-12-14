#!/bin/bash

# python
python3.10 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel cython
pip install setuptools==69.5.1
pip install packaging
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_cuda11.txt

# apex install
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout $apex_commit
pip install . -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam --group_norm"
cd ../

# transformer engine install
PATH=/usr/local/cuda/bin:$PATH pip install git+https://github.com/NVIDIA/TransformerEngine.git@c81733f1032a56a817b594c8971a738108ded7d0 --no-cache-dir
