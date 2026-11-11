# simulador_nuevo

Simulador de videos de fluorescencia escrito a partir de sus tests. Usa la
misma optica, camara y rangos que `sim_imagenesPSF.py`, y es independiente de
el. La unica importacion de fuera de esta carpeta es `geometria_espejo.py` (en
la raiz del repo, compartida con `simulador_psf`): `simulador.py` agrega la
raiz al FINAL de `sys.path` (como `localizacion/data/simulacion.py` agrega esta
carpeta), asi que hay que conservar la estructura del repo.

```
simulador_nuevo/
  simulador.py          el simulador (Config, BancoPSF, generar_video, generar_dataset, diagnosticos)
  tests/                tests con asserts (pasa / falla)
    correr_tests.py     corre todos los tests, sin necesidad de pytest
  checks/               diagnosticos con figuras (fig_*.png), como checks/ de la raiz
    run_all_checks.py
  salida/               lo que escribe simulador.py (ignorado por git)
```

```bash
PY=~/miniforge3/envs/tesis/bin/python
$PY simulador_nuevo/simulador.py                     # dataset -> simulador_nuevo/salida/
$PY simulador_nuevo/tests/correr_tests.py            # 81 tests, ~9 s
$PY simulador_nuevo/tests/correr_tests.py psf        # solo los tests cuyo nombre contiene "psf"
$PY simulador_nuevo/checks/run_all_checks.py         # ~2 min (check_crlb hace ajustes MLE)
```

## Que se verifica

| Archivo | Que verifica |
|---|---|
| `test_config.py` | pixel en la muestra, Nyquist, rechazo de parametros invalidos, ida y vuelta a JSON |
| `test_psf.py` | normalizacion y no negatividad, recorte automatico, simetrias, PSF en foco = Airy, desenfoque en el eje vs integral teorica, interpolacion en z, integracion sobre el pixel vs sobremuestreo 5x5, DOF de la esferica, giro del corkscrew |
| `test_colocar.py` | ejes x/y, centro de pixel en enteros, conservacion de fotones, shift subpixel vs PSF exacta, traslaciones enteras, linealidad, bordes |
| `test_camara.py` | media y varianza en ADU (Poisson + lectura + cuantizacion), curva de transferencia de fotones, saturacion, conversion inversa |
| `test_fondo.py` | fondo uniforme / no uniforme acotado y suave |
| `test_difusion.py` | paredes reflectantes, MSD = 2Dt por eje, incrementos gaussianos independientes, z uniforme, desenfoque de movimiento (MSD con correccion t_e(n^2-1)/(3n^2)) |
| `test_video.py` | forma del video, columnas y filas del GT, en_fov, imagen limpia reconstruible desde el GT, residuo de ruido ~ N(0,1), semilla, componentes independientes, prefijo temporal, escritura a disco |
| `test_espejo.py` | validacion y JSON de los campos del espejo, misma geometria que `simulador_psf`, banco ampliado con los planos de `z_rango_nm` intactos, ningun real cruza el espejo (tambien en los subpasos), `dist_max_espejo_um`, espejo lejano = sin espejo, GT del reflejo, desenfoque ligado, imagen limpia reconstruible (con subpasos y desde el GT), el espejo no cambia los otros flujos, regeneracion por video, convencion (reflejo del lado correcto, `anillos_hacia_objetivo` solo cambia la PSF, `config.json` de la convencion anterior rechazado), `ratio_reflejo = 0` sin luz del reflejo, espejo apagado |
| `test_celula.py` | celula (`usar_celula`): validacion y JSON, tangente al espejo, posiciones uniformes en el volumen con los mismos sorteos que sin celula, reflexion especular exacta, ningun emisor sale ni cruza el espejo (tambien en los subpasos), MSD que se satura en 6 R^2 / 5, reflejos dentro de la celula virtual, `en_rango_z` y banco, celula apagada |
| `test_recorte.py` | recorte de pupila del reflejo: fraccion = integral de la mascara (y la geometrica), beta = 0 -> la mitad, beta >= 55.6 -> nada, fraccion(-beta) = 1 - fraccion(beta), PSF sin renormalizar (suma la fraccion), phi = 90 = rotacion exacta, misma geometria con las dos variantes de anillos, corrimiento del centroide hacia +m con z' < 0 (optica geometrica), real/GT/flujos sin cambio, imagen reconstruible, sin recorte efectivo bit a bit; recorte del real: pierde lo mismo que el reflejo a 45 grados (4.67%), GT sin cambio, imagen reconstruible, bit a bit sin efecto |
| `test_regresion.py` | con el espejo apagado, salidas bit a bit iguales a las de antes del espejo (`referencia_regresion.json`; se salta si cambia la version de numpy; `--generar` la reescribe) |
| `test_diagnosticos.py` | SNR del pixel pico vs simulacion, escalamiento de la CRLB; CRLB 3D (`crlb_3d`, z desconocido) vs la PSF exacta de `psf_directa`, vs la 2D y escalamiento con N |

## Diferencias con sim_imagenesPSF.py

- **Integracion sobre el pixel** (`integrar_pixel=True`): la PSF se integra sobre el area del pixel en vez de muestrearse en su centro. Ensancha un poco la PSF; con `False` se recupera el muestreo puntual.
- **Recorte automatico de la PSF** (`K_psf=None`): se usa el menor K que pierde como mucho `energia_max_recortada` (2%). Queda K = 128 con la esferica (igual que antes), 64 sin mascara y 144 con el corkscrew (antes 48, que perdia ~7%).
- **Desenfoque de movimiento** opcional (`t_exp_s`, `n_subpasos`; por defecto apagado). El GT es la posicion media durante la exposicion.
- **Convencion de pixel**: el centro del pixel (fila i, col j) es (x=j, y=i) y el FOV es [-0.5, W-0.5) x [-0.5, H-0.5). Antes `en_fov` usaba [0, W).
- **Aleatoriedad por video y por componente**: cada video se regenera solo, y cambiar ruido, fondo o fotones no cambia las trayectorias. Por eso, con la misma semilla, los videos no son iguales a los de `sim_imagenesPSF.py`.
- La salida incluye `config.json` (Config + K usado), y hay diagnosticos nuevos: `crlb` (z conocido), `crlb_3d` (z desconocido), `modelo_roi`, `en_fov`, `psf_directa`.

## Espejo (`usar_espejo=True`)

Mismo modelo y convencion que `simulador_psf` (README de la raiz, secciones
"Convencion de z" y Espejo: +z hacia el objetivo, angulo desde el eje
optico), con la geometria de `geometria_espejo.py` y los mismos nombres de
campos (`angulo_espejo_eje_deg`, `orientacion_espejo_deg`, `espejo_punto_px`,
`ratio_reflejo`, `z_ref_espejo_nm`, `dist_max_espejo_um`, `z_banco_nm`; y
`anillos_hacia_objetivo`, de la PSF):

- Solo se simulan las trayectorias reales. Con espejo se integran subpaso a
  subpaso en nm, plegando en las paredes en z y en el espejo, con los mismos
  sorteos del flujo `trayectorias`; con `dist_max_espejo_um` los sorteos de x
  e y se reescalan a la franja cercana al espejo.
- Cada subposicion del reflejo es el reflejo de la del real, asi que el
  desenfoque de movimiento queda ligado; se dibuja con
  `ratio_reflejo * fotones / n_subpasos` (`ratio_reflejo` en [0, 1]; con 0 el
  reflejo queda en el GT pero sin luz, como control). El GT agrega `COLUMNAS_REFLEJO`
  (`x_reflejo_px, y_reflejo_px, z_reflejo_nm, photons_reflejo, reflejo_en_fov,
  reflejo_en_banco`); la posicion del reflejo es la media de sus subposiciones.
- El banco se amplia (`rango_banco_nm`) para cubrir las z' de la caja inicial,
  con los mismos planos que sin espejo dentro de `z_rango_nm`; `z_banco_nm`
  fija a mano el rango a cubrir. Un reflejo fuera del banco no se dibuja
  (`reflejo_en_banco = False`). `K_psf=None` elige el recorte sobre todo el
  banco: a 45 grados los reflejos quedan hasta z' = -4.6 um (mas lejos del
  objetivo); con la esferica, `anillos_hacia_objetivo = False` deja K = 128,
  pero con `True` ese lado es el de la PSF ancha y K sube a 208, y cada
  muestra cuesta el doble.
- Recorte de pupila del reflejo (`recorte_pupila_reflejo=True`, apagado por
  defecto): el rayo de direccion w (+z hacia el objetivo) que parece salir del
  reflejo virtual existe solo si cruza el plano del espejo hacia la muestra,
  w . n > 0 con n = cos(beta) m + sin(beta) z (espejo infinito; un borde finito
  recortaria mas). La mascara (`mascara_recorte`) es la misma para todo el
  campo, asi que `PSFRecortada` calcula la PSF de cada subposicion del reflejo
  con una FFT de la pupila recortada (z' exacta, desplazamiento subpixel en la
  pupila; ~2 ms por PSF con `M_pupila` 256) y la coloca con el mismo recorte K
  que el banco. No se renormaliza: la PSF suma la fraccion de la pupila que
  pasa (`PSFRecortada.fraccion`: 95.3% a 45 grados, 50% a 0, 100% desde
  arcsin(NA / n) = 55.6), asi que `ratio_reflejo` queda como la reflectividad
  y la luz del reflejo en la imagen es `ratio_reflejo * fraccion * fotones`.
  El GT no cambia (`photons_reflejo` sigue siendo `ratio_reflejo * photons`),
  el real tampoco, y si pasa toda la pupila se usa el banco (bit a bit igual a
  sin recorte). Con `anillos_hacia_objetivo=True` la mascara se toma en -f
  para que el espejo recorte el mismo lado fisico (con la mascara "ninguna"
  las dos variantes dan la misma PSF recortada).
- Recorte de pupila del real (`recorte_pupila_real=True`, apagado por
  defecto): los rayos directos del real con w . n < 0 chocan con el espejo
  antes del objetivo (son los que forman el reflejo), asi que el real pierde la
  misma parte de la pupila, con la misma mascara y la misma `PSFRecortada`
  (4.7% a 45 grados, igual que el reflejo). Supone el espejo infinito tambien
  por encima del plano focal: el montaje es un portaobjetos espejado de varios
  mm. `photons` del GT sigue siendo lo emitido. Con los dos recortes una
  muestra de `celula.yaml` cuesta ~20 ms con 8 workers.
- No consume numeros aleatorios: n, D, fotones y fondo son los mismos con y
  sin espejo. Con el espejo apagado sus campos no se escriben en
  `config.json` (`anillos_hacia_objetivo`, que es de la PSF, si).
- `config.json` lleva `convencion_z`; `cargar_config` rechaza los de la
  convencion anterior. Con el espejo apagado las imagenes son bit a bit las
  de la convencion anterior y el GT solo cambia el signo de z
  (`tests/test_convencion_z.py`, en la raiz).

## Celula (`usar_celula=True`)

Los emisores estan dentro de una celula esferica (levadura), con la geometria
de `geometria_celula.py` (en la raiz, compartida):

- `diametro_celula_um` (5 por defecto), `altura_celula_nm` (z del centro
  respecto del foco, + hacia el objetivo; 0 por defecto, pendiente de
  confirmar) y `celula_centro_px`. Con espejo la celula lo toca: su centro esta
  a R del plano, y `celula_centro_px` solo fija su posicion a lo largo de la
  linea del espejo. Una sola celula por imagen.
- Posiciones iniciales uniformes en el volumen, con los mismos 3 sorteos por
  emisor que sin celula (n, D, fotones y fondo no cambian). Browniano
  integrado subpaso a subpaso con reflexion especular exacta en la
  superficie; como la celula toca el espejo, ningun emisor lo cruza.
- `z_rango_nm` deja de limitar a los emisores: solo define el rango axial de
  la red y la base del banco, que se amplia para cubrir la celula y la celula
  virtual. El GT agrega `en_rango_z` (z dentro de `z_rango_nm`), que
  `localizacion` trata como `en_fov = False`.
- Con la celula apagada sus campos no se escriben en `config.json` y todo es
  identico bit a bit a antes.
