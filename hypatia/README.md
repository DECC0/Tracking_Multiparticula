# Entrenamiento en Hypatia (SLURM)

Todos los comandos se corren desde la raiz del repo en Hypatia, salvo el paso 6.

1. **Clonar** la rama `prueba`:
   ```bash
   git clone -b prueba https://github.com/DECC0/Tracking_Multiparticula.git
   cd Tracking_Multiparticula
   ```

2. **Instalar el entorno** `tracking` (una vez, en el nodo de login):
   ```bash
   source hypatia/instalar_entorno.sh
   ```

3. **Probar en un nodo GPU** antes de lanzar el job largo:
   ```bash
   salloc -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=1:00:00
   module load anaconda && source activate tracking
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

5. **Lanzar el entrenamiento** (`configs/base.yaml`, 30 epocas, 8 workers; simula los datos on-the-fly, no hay que copiar datos):
   ```bash
   sbatch hypatia/train.sbatch
   squeue -u $USER                    # estado del job
   tail -f hypatia/logs/train.o<JOBID>
   ```
   Si se acaba el tiempo, se continua desde `ultimo.pt` con
   `python -u localizacion/train.py --reanudar localizacion/corridas/hypatia_base_<fecha-hora>` (cambiando la linea `python` del `.sbatch`).

6. **Bajar la corrida** al computador local (desde la raiz del repo local):
   ```bash
   rsync -avz <usuario>@hypatia.uniandes.edu.co:<ruta>/Tracking_Multiparticula/localizacion/corridas/hypatia_base_<fecha-hora> localizacion/corridas/
   ```
