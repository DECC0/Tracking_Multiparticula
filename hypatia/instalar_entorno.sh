#!/bin/bash
# Crea el entorno virtual ~/envs/tracking en Hypatia. Se corre UNA vez en el
# nodo de login (necesita internet), NO con sbatch:
#     source hypatia/instalar_entorno.sh
#
# Usa el modulo python/3.11 del cluster + venv, no el modulo anaconda: su conda
# 4.12 muere por memoria en el login al resolver conda-forge.
#
# Paquetes: los que importan localizacion/ y simulador_nuevo/ (numpy, scipy,
# pandas, matplotlib, pyyaml, tifffile, torch), con las versiones del entorno
# local "tesis" salvo torch (ver abajo). Se detiene en el primer paso que
# falle sin cerrar la shell (por eso "|| return 1" y no "set -e").

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "Correr con: source hypatia/instalar_entorno.sh" >&2
    exit 1
fi

VENV=~/envs/tracking

module purge || return 1
module load python/3.11 || return 1
python3 -c "import sys; sys.exit(sys.version_info[:2] != (3, 11))" \
    || { echo "python3 no es 3.11: $(python3 --version 2>&1)" >&2; return 1; }

if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV" || return 1
fi
source "$VENV/bin/activate" || return 1
# pip solo con el python del venv (no con el del modulo)
[ "$(which python)" = "$VENV/bin/python" ] \
    || { echo "python no es el del venv: $(which python)" >&2; return 1; }
python -c "import sys; sys.exit(sys.version_info[:2] != (3, 11))" \
    || { echo "el venv $VENV no es de Python 3.11: $(python --version 2>&1)" >&2; return 1; }

# el /tmp del login es pequeno: pip deja ahi los wheels (torch pesa varios GB) y
# la instalacion queda incompleta. Se usa ~/tmp y se borra al final si no existia
[ -d ~/tmp ] && TMP_EXISTIA=1 || TMP_EXISTIA=0
mkdir -p ~/tmp && export TMPDIR=~/tmp || return 1

pip install --no-cache-dir --upgrade pip || return 1
# torch 2.7.1+cu118, no el 2.11.0+cu128 del entorno local: el driver de las GPU de
# Hypatia solo soporta CUDA 11.8 (torch reporta driver version 11080) y con cu128
# torch no ve la GPU. torchvision no lo usa el codigo, se instala por compatibilidad
pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118 || return 1
pip install --no-cache-dir numpy==2.4.6 scipy==1.17.1 pandas==3.0.5 matplotlib==3.11.1 pyyaml==6.0.3 tifffile==2026.3.3 || return 1

# en el nodo de login no hay GPU: cuda False aqui es normal
python -c "import torch, numpy, scipy, pandas, matplotlib, yaml, tifffile; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || return 1

unset TMPDIR
[ "$TMP_EXISTIA" = 0 ] && rm -rf ~/tmp
unset TMP_EXISTIA
