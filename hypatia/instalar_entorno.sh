#!/bin/bash
# Crea el entorno conda "tracking" en Hypatia. Se corre UNA vez en el nodo de
# login (necesita internet), NO con sbatch:
#     source hypatia/instalar_entorno.sh
#
# Paquetes: los que importan localizacion/ y simulador_nuevo/ (numpy, scipy,
# pandas, matplotlib, pyyaml, tifffile, torch), con las versiones del entorno
# local "tesis". No se usa environment.yml: su torch +cu128 (bloque pip) no se
# resuelve sin el indice de PyTorch.

module load anaconda
conda create -n tracking python=3.11 -c conda-forge -y
source activate tracking

# torch con CUDA 12.8 (torchvision no lo usa el codigo, se instala por compatibilidad)
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==2.4.6 scipy==1.17.1 pandas==3.0.5 matplotlib==3.11.1 pyyaml==6.0.3 tifffile==2026.3.3

# en el nodo de login no hay GPU: cuda False aqui es normal
python -c "import torch, numpy, scipy, pandas, matplotlib, yaml, tifffile; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
