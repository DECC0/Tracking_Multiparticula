#!/bin/bash
# Lanza un entrenamiento con espejo (hypatia/espejo.sbatch) con los recursos y
# rutas de abajo. Desde la raiz del repo en Hypatia:
#     bash hypatia/lanzar_espejo.sh 1          # signo_z = +1
#     bash hypatia/lanzar_espejo.sh -1         # signo_z = -1
#     EPOCAS=100 TIEMPO=0-08:00:00 bash hypatia/lanzar_espejo.sh 1    # corrida completa
# Cualquier variable de abajo se puede cambiar asi, en la linea de comandos.

# ---------------- variables ----------------
SIGNO_Z=${1:-1}                                   # +1 o -1 (Config.signo_z del simulador)
EPOCAS=${EPOCAS:-3}                               # piloto: pocas epocas (espejo.yaml trae 100)
TIEMPO=${TIEMPO:-0-01:30:00}                      # limite del job (d-hh:mm:ss)
MEMORIA=${MEMORIA:-24G}                           # RAM del job
CPUS=${CPUS:-4}                                   # = workers del DataLoader (simulan on-the-fly)
CONFIG=${CONFIG:-localizacion/configs/espejo.yaml}
VENV=${VENV:-$HOME/envs/tracking}                 # venv de hypatia/instalar_entorno.sh (torch 2.7.1+cu118)
DATOS=${DATOS:-localizacion/datos}                # donde queda el test generado (test_espejo_sz<signo>)
N_TEST=${N_TEST:-2000}                            # muestras del test
EXTRA_SET=${EXTRA_SET:-}                          # overrides extra de train.py, p. ej. "datos.n_train=5000"
                                                  # (sin comas: sbatch --export las usa de separador)
# -------------------------------------------

if [ "$SIGNO_Z" != "1" ] && [ "$SIGNO_Z" != "-1" ] && [ "$SIGNO_Z" != "+1" ]; then
    echo "SIGNO_Z debe ser 1 o -1 (es $SIGNO_Z)" >&2
    exit 1
fi
SIGNO_Z=$((SIGNO_Z))
if [ ! -f hypatia/espejo.sbatch ]; then
    echo "Lanzar desde la raiz del repo" >&2
    exit 1
fi
mkdir -p hypatia/logs                             # sin ella SLURM falla sin escribir nada

sbatch --job-name="espejo_sz$(printf %+d $SIGNO_Z)" --time="$TIEMPO" --mem="$MEMORIA" --cpus-per-task="$CPUS" \
    --export=ALL,SIGNO_Z="$SIGNO_Z",EPOCAS="$EPOCAS",CONFIG="$CONFIG",VENV="$VENV",DATOS="$DATOS",N_TEST="$N_TEST",EXTRA_SET="$EXTRA_SET" \
    hypatia/espejo.sbatch
