#!/bin/bash
# Uso (desde la raiz del repo):
#     sbatch hypatia/train.sbatch                                   # localizacion/configs/base.yaml
#     sbatch hypatia/train.sbatch localizacion/configs/mezcla.yaml

# ###### Zona de Parametros de solicitud de recursos a SLURM ############################
#
#SBATCH --job-name=tracking_train          #Nombre del job
#SBATCH -p gpu                             #Cola a usar (particion con GPU)
#SBATCH -N 1                               #Nodos requeridos
#SBATCH -n 1                               #Tasks paralelos (un solo proceso de entrenamiento)
#SBATCH --cpus-per-task=8                  #Cores por task: 8 workers del DataLoader simulan on-the-fly
#SBATCH --gres=gpu:1                       #GPUs requeridas
#SBATCH --mem=32G                          #Memoria RAM del job
#SBATCH --time=0-12:00:00                  #Tiempo maximo de corrida (d-hh:mm:ss)
#SBATCH --mail-user=de.cardenasc@uniandes.edu.co
#SBATCH --mail-type=ALL                    #Correo al iniciar, terminar o fallar
#SBATCH -o hypatia/logs/train.o%j          #Salida estandar (relativa a la carpeta desde donde se hace sbatch)
#SBATCH -e hypatia/logs/train.e%j          #Salida de errores
#
########################################################################################

# ################## Zona Carga de Modulos ############################################
module purge
module load anaconda
source activate tracking

########################################################################################


# ###### Zona de Ejecucion de codigo y comandos a ejecutar secuencialmente #############
# sbatch se lanza desde la raiz del repo (ahi deben existir hypatia/logs/ y localizacion/)
cd "$SLURM_SUBMIT_DIR"
if [ ! -f localizacion/train.py ]; then
    echo "Lanzar con sbatch desde la raiz del repo (no se encontro localizacion/train.py en $PWD)" >&2
    exit 1
fi

CONFIG=${1:-localizacion/configs/base.yaml}
if [ ! -f "$CONFIG" ]; then
    echo "No existe la config $CONFIG" >&2
    exit 1
fi
# nombre de la corrida: el de la config, con prefijo hypatia_ para distinguirla de las locales
NOMBRE=hypatia_$(python -c "import sys, yaml; c = yaml.safe_load(open(sys.argv[1])) or {}; print((c.get('entrenamiento') or {}).get('nombre', 'corrida'))" "$CONFIG")

host=`/bin/hostname`
date=`/bin/date`
echo "Corri en la maquina: "$host
echo "Corri el: "$date
echo "Config: $CONFIG  ->  corrida $NOMBRE"
nvidia-smi

# la corrida queda en localizacion/corridas/<NOMBRE>_<fecha-hora>/; las epocas son las de la config
python -u localizacion/train.py --config "$CONFIG" \
    --set entrenamiento.workers=8 entrenamiento.nombre=$NOMBRE

echo "Termino el: "`/bin/date`

########################################################################################
