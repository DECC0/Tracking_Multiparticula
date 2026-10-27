# Entrenamiento en Hypatia (SLURM)

Todos los comandos se corren desde la raiz del repo en Hypatia, salvo el paso 6.

1. **Clonar** la rama `prueba`:
   ```bash
   git clone -b prueba https://github.com/DECC0/Tracking_Multiparticula.git
   cd Tracking_Multiparticula
   ```

2. **Instalar el entorno** (una vez, en el nodo de login): crea el venv `~/envs/tracking` con el modulo `python/3.11` e instala los paquetes. Se corre con `source`; si un paso falla se detiene sin cerrar la shell:
   ```bash
   source hypatia/instalar_entorno.sh
   ```
   No se usa el modulo `anaconda`: su conda 4.12 muere por memoria en el login al resolver conda-forge (y pip terminaba instalando en su Python 3.9).
   El driver de las GPU de Hypatia es CUDA 11.8, por eso el cluster usa torch 2.7.1+cu118 mientras el entorno local usa 2.11.0+cu128. El script instala en `~/tmp` (`TMPDIR`) porque el `/tmp` del login es pequeno y torch quedaba incompleto.

3. **Probar en un nodo GPU** antes de lanzar el job largo:
   ```bash
   salloc -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=1:00:00
   module purge && module load python/3.11 && source ~/envs/tracking/bin/activate
   nvidia-smi
   python localizacion/train.py --config localizacion/configs/smoke.yaml --set entrenamiento.dispositivo=cuda
   python localizacion/tests/correr_tests.py
   python simulador_nuevo/tests/correr_tests.py
   exit
   ```
   Si `hostname` muestra que `salloc` dejo la shell en el nodo de login, antepon `srun` a cada comando.

4. **Crear la carpeta de logs** (SLURM no la crea y sin ella el job falla sin escribir nada):
   ```bash
   mkdir -p hypatia/logs
   ```

5. **Lanzar el entrenamiento** con una config (por defecto `configs/base.yaml`; 4 workers; simula los datos on-the-fly, no hay que copiar datos). La corrida se llama `hypatia_<entrenamiento.nombre de la config>`:
   ```bash
   sbatch hypatia/train.sbatch                                   # base.yaml  -> hypatia_base
   sbatch hypatia/train.sbatch localizacion/configs/mezcla.yaml  # mezcla.yaml -> hypatia_mezcla
   sbatch hypatia/train.sbatch localizacion/configs/mezcla_renovar.yaml  # train nuevo en cada epoca, 50 epocas
   sbatch hypatia/train.sbatch localizacion/configs/mezcla_renovar_largo.yaml  # igual, 100 epocas
   sbatch hypatia/train.sbatch localizacion/configs/mezcla_renovar_fija.yaml   # igual a _largo con normalizacion fija
   squeue -u $USER                    # estado del job
   tail -f hypatia/logs/train.o<JOBID>
   ```
   Si se acaba el tiempo, se continua desde `ultimo.pt` con
   `python -u localizacion/train.py --reanudar localizacion/corridas/hypatia_<nombre>_<fecha-hora>` (cambiando la linea `python` del `.sbatch`).

6. **Bajar la corrida** al computador local (desde la raiz del repo local):
   ```bash
   rsync -avz <usuario>@hypatia.uniandes.edu.co:<ruta>/Tracking_Multiparticula/localizacion/corridas/hypatia_<nombre>_<fecha-hora> localizacion/corridas/
   ```
