#!/bin/bash
# Lanza un entrenamiento con espejo (hypatia/espejo.sbatch) con los recursos y
# rutas de abajo. Desde la raiz del repo en Hypatia:
#     bash hypatia/lanzar_espejo.sh lejos      # anillos de la PSF lejos del objetivo (anillos_hacia_objetivo false)
#     bash hypatia/lanzar_espejo.sh cerca      # anillos hacia el objetivo (anillos_hacia_objetivo true)
#     EPOCAS=100 TIEMPO=0-09:00:00 bash hypatia/lanzar_espejo.sh lejos    # corrida completa
# Convencion: +z hacia el objetivo, espejo a 45 grados del eje optico (configs/espejo.yaml).
# Cualquier variable de abajo se puede cambiar asi, en la linea de comandos.

# ---------------- variables ----------------
ANILLOS=${1:-lejos}                               # lejos | cerca (Config.anillos_hacia_objetivo false | true)
EPOCAS=${EPOCAS:-3}                               # piloto: pocas epocas (espejo.yaml trae 100)
TIEMPO=${TIEMPO:-0-01:30:00}                      # limite del job (d-hh:mm:ss)
MEMORIA=${MEMORIA:-24G}                           # RAM del job
CPUS=${CPUS:-4}                                   # = workers del DataLoader (simulan on-the-fly)
CONFIG=${CONFIG:-localizacion/configs/espejo.yaml}
VENV=${VENV:-$HOME/envs/tracking}                 # venv de hypatia/instalar_entorno.sh (torch 2.7.1+cu118)
DATOS=${DATOS:-localizacion/datos}                # donde queda el test generado (test_espejo_b45_<anillos>)
N_TEST=${N_TEST:-2000}                            # muestras del test
EXTRA_SET=${EXTRA_SET:-}                          # overrides extra de train.py, p. ej. "datos.n_train=5000"
                                                  # (sin comas: sbatch --export las usa de separador)
# -------------------------------------------

if [ "$ANILLOS" != "lejos" ] && [ "$ANILLOS" != "cerca" ]; then
    echo "el argumento debe ser lejos o cerca (es $ANILLOS)" >&2
    exit 1
fi
if [ ! -f hypatia/espejo.sbatch ]; then
    echo "Lanzar desde la raiz del repo" >&2
    exit 1
fi
mkdir -p hypatia/logs                             # sin ella SLURM falla sin escribir nada

sbatch --job-name="espejo_b45_$ANILLOS" --time="$TIEMPO" --mem="$MEMORIA" --cpus-per-task="$CPUS" \
    --export=ALL,ANILLOS="$ANILLOS",EPOCAS="$EPOCAS",CONFIG="$CONFIG",VENV="$VENV",DATOS="$DATOS",N_TEST="$N_TEST",EXTRA_SET="$EXTRA_SET" \
    hypatia/espejo.sbatch
