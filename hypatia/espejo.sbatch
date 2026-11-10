#!/bin/bash
# Entrenamiento con el espejo (localizacion/configs/espejo.yaml) en la particion gpu.
# No se lanza directo: hypatia/lanzar_espejo.sh fija tiempo, memoria, CPUs y las
# variables de abajo (ANILLOS, EPOCAS, ...) y llama a sbatch con este archivo.
#
# Pasos del job (desde la raiz del repo):
#   1. genera el test con espejo de esta PSF (localizacion/generar_espejo.py), si no existe
#   2. entrena con espejo.yaml (datos on-the-fly en los workers, train nuevo en cada epoca)
#   3. evalua mejor.pt sobre ese test (evaluate.py)

# ###### Zona de Parametros de solicitud de recursos a SLURM ############################
# (valores por defecto; lanzar_espejo.sh los reemplaza con --time, --mem, --cpus-per-task)
#SBATCH --job-name=espejo
#SBATCH -p gpu                             #Particion con GPU (Quadro RTX 6000)
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=0-01:00:00
#SBATCH --mail-user=de.cardenasc@uniandes.edu.co
#SBATCH --mail-type=ALL
#SBATCH -o hypatia/logs/espejo.o%j
#SBATCH -e hypatia/logs/espejo.e%j
########################################################################################

# variables (las exporta lanzar_espejo.sh; aqui sus valores por defecto)
ANILLOS=${ANILLOS:-lejos}                 # lejos | cerca (Config.anillos_hacia_objetivo false | true)
PREFIJO=${PREFIJO:-espejo}                # espejo | celula (configs/celula.yaml)
EPOCAS=${EPOCAS:-3}
CONFIG=${CONFIG:-localizacion/configs/espejo.yaml}
VENV=${VENV:-$HOME/envs/tracking}
DATOS=${DATOS:-localizacion/datos}
N_TEST=${N_TEST:-2000}
EXTRA_SET=${EXTRA_SET:-}

module purge
module load python/3.11
source "$VENV/bin/activate" || exit 1

cd "$SLURM_SUBMIT_DIR"
if [ ! -f localizacion/train.py ]; then
    echo "Lanzar desde la raiz del repo (no se encontro localizacion/train.py en $PWD)" >&2
    exit 1
fi
set -e

if [ "$ANILLOS" = "cerca" ]; then BANDERA=true; OPCION=--anillos-hacia-objetivo; else BANDERA=false; OPCION=; fi
NOMBRE=hypatia_${PREFIJO}_b45_${ANILLOS}
TEST=$DATOS/test_${PREFIJO}_b45_${ANILLOS}
WORKERS=${SLURM_CPUS_PER_TASK:-4}

echo "Corri en la maquina: $(/bin/hostname)  el: $(/bin/date)"
echo "config $CONFIG, anillos $ANILLOS (anillos_hacia_objetivo $BANDERA), $EPOCAS epocas, $WORKERS workers -> corrida $NOMBRE"
nvidia-smi

# 1. test con espejo (mismo para las corridas de esta PSF; se genera una vez)
if [ ! -f "$TEST/imagenes.npy" ]; then
    python -u localizacion/generar_espejo.py --config "$CONFIG" $OPCION --split test \
        --n "$N_TEST" --salida "$TEST"
fi

# 2. entrenamiento: la carpeta de la corrida es localizacion/corridas/<NOMBRE>_<fecha-hora>/
python -u localizacion/train.py --config "$CONFIG" \
    --set datos.simulador.anillos_hacia_objetivo=$BANDERA entrenamiento.epocas=$EPOCAS \
          entrenamiento.workers=$WORKERS entrenamiento.nombre=$NOMBRE $EXTRA_SET
CORRIDA=$(ls -td localizacion/corridas/${NOMBRE}_*/ | head -1)

# 3. evaluacion de la mejor epoca sobre el test de esta PSF
python -u localizacion/evaluate.py "$CORRIDA/mejor.pt" "$TEST" --figuras 6

echo "Termino el: $(/bin/date)  -> $CORRIDA"
