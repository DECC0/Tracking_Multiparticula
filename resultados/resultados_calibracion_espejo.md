# Calibración del espejo con un z-stack de beads (propuesta de análisis)

Propuesta, sin datos todavía. Convención del repo: +z hacia el objetivo y el ángulo del espejo β desde el eje óptico (montaje: 45°). El objetivo es medir con beads junto al espejo:

1. `anillos_hacia_objetivo`: de qué lado del foco gana anillos la PSF esférica;
2. el ángulo β;
3. la línea del espejo (φ y `espejo_punto_px`);
4. `ratio_reflejo`;
5. la PSF empírica del reflejo, con su recorte de pupila.

## Qué se mide

Se adquiere un z-stack moviendo la platina (muestra y espejo juntos) con beads fluorescentes fijas a distintas distancias d del espejo. Al mover la platina Δ, el real y su imagen virtual se trasladan igual, así que cada uno pasa por foco en una posición distinta de la platina:

- real en foco en $s_r = -z_r$;
- reflejo en foco en $s_f = -z_f$;
- con $z_f - z_r = -2d\sin\beta$: el reflejo entra en foco $2d\sin\beta$ más lejos del objetivo.

Por cada bead se miden:
- la posición de platina del foco del real y la del reflejo;
- las posiciones laterales del real y del reflejo, cada una en su propio foco;
- los fotones totales de cada uno;
- el stack completo de imágenes de cada uno alrededor de su foco.

## 1. `anillos_hacia_objetivo`

El reflejo sirve de referencia de dirección, y no hace falta conocer el signo de la platina. Como el reflejo está siempre más lejos del objetivo, el sentido en que hay que mover la platina para pasar del foco del real al del reflejo es el de alejarse del objetivo. Con eso:

- **Si la PSF del real gana anillos en ese mismo sentido**, `anillos_hacia_objetivo = False`; si los gana en el sentido contrario, `True`.
- **Confirmación:** comparar el stack del real con los stacks simulados de las dos variantes (`BancoPSF` con `anillos_hacia_objetivo` False y True). Se correlaciona cada plano con el simulado a ±z y se elige la variante de mayor correlación total. Basta con una bead lejos del espejo (d > 5 µm), para que el reflejo no se superponga.

El frame real que ya tenemos (reflejo compacto) apunta a `True`, pero el recorte de pupila también deforma el reflejo. Por eso el criterio tiene que usar la PSF del **real**, que no está recortada.

## 2. Ángulo β

Para cada bead, $\tan\beta = -\Delta z / \Delta_{lat}$, con $\Delta z = z_f - z_r$ (de las posiciones de foco) y $\Delta_{lat}$ la separación lateral real–reflejo, medida en el foco de cada uno. En el foco del reflejo su desenfoque es nulo y el corrimiento del centroide por el recorte es mínimo.

Con varias beads conviene un ajuste lineal $\Delta z_i = -\tan\beta\,\Delta_{lat,i} + c$. La ordenada c absorbe un error de cero común. Incertidumbre:

- **Una bead:** $\delta\beta \approx \sqrt{\sigma_{\Delta z}^2\cos^2\beta + \sigma_{\Delta lat}^2\sin^2\beta}\,/\,(2d)$. Con σ_Δz = 28 nm (20 nm por foco) y σ_Δlat = 10 nm, a 45° queda δβ ≈ 21 nm / 2d. Bajar de 0.1° (1.75 mrad) con una sola bead exige d ≥ 6 µm.
- **Ajuste con N beads repartidas en d = 1–8 µm** ($\Delta_{lat}$ de 1.4 a 11.3 µm): $\sigma_{pendiente} \approx 30\ \text{nm}/\sqrt{8.2\,N}$ µm y $\delta\beta = \sigma_{pendiente}\cos^2\beta$. Bajar de 0.1° pide **N ≥ 10**.
- **Propuesta:** 15 beads repartidas de forma pareja entre d = 1 y 8 µm, en dos o tres orientaciones de la línea del espejo dentro del campo, para separar β de una posible inclinación del plano focal.

Por qué importa la precisión (verificado en `resultados_temporal_espejo.md`): un error δβ sesga la z triangulada en ≈ 2·m·δβ. Con ±0.5° (el redondeo del valor medido) son 35 nm a 2 µm de la línea, del orden de la ganancia que aporta el reflejo.

## 3. Línea del espejo (φ y `espejo_punto_px`)

El punto medio $(p + p')/2$ de cada par está sobre el plano del espejo, y $p' - p$ es paralelo a su normal $n = \cos\beta\,\hat m + \sin\beta\,\hat z$. Ajustar el plano a los N puntos medios (mínimos cuadrados totales, con la normal restringida a ser paralela a los $p' - p$) da β, φ y $p_0$ a la vez. La línea del espejo en el plano focal es la intersección con z = 0, tomando como z = 0 la posición de platina de referencia de la adquisición.

Tolerancia: un error δ en la posición de la línea sesga la z triangulada en δ·tan β, es decir 34 nm por cada 50 nm a 45°. Por eso hace falta la línea a ≲ 15 nm, lo que con N = 15 y ~10 nm por punto medio es alcanzable.

## 4. `ratio_reflejo`

Se mide el cociente de fotones totales reflejo/real, cada uno en su foco, con el fondo restado sobre una apertura grande (≥ 3 µm de radio: la PSF esférica tiene halo).

- **Valor esperado:** reflectividad × (1 − recorte). El recorte a 45° es ~4.7–5.9% con un espejo infinito (cálculo geométrico, abajo).
- **Dependencia con d y con φ:** comprobarla. Si el cociente baja con d, el espejo finito recorta más que el modelo infinito.

## 5. PSF empírica del reflejo

- **Banco empírico:** registrar las imágenes del reflejo por su centroide en cada plano y ordenarlas por desenfoque relativo a su foco.
- **Comparación:** contrastar con la PSF del real al mismo desenfoque. La asimetría a lo largo de $\hat m$ y el corrimiento del centroide con el desenfoque miden el recorte.
- **Predicción geométrica (espejo infinito, β = 45°):** el centroide se corre 57–73 nm por µm de desenfoque, hacia +m.
- **Uso:** con eso se puede reemplazar el banco del reflejo en el simulador por uno empírico (o por el modelo de pupila recortada) en vez de la PSF completa.

## Adquisición

- **Paso en z:** 50 nm. Con la máscara esférica la profundidad de campo es extendida y el foco por intensidad pico es ancho. Por eso el foco se ubica por correlación con el banco (o con el stack del propio real), no por el máximo. Con 50 nm y un ajuste local se esperan ~20 nm por foco.
- **Si el montaje permite quitar la máscara de fase**, conviene una adquisición sin ella para los focos (profundidad de campo ~0.6 µm, focos más nítidos) y otra con ella para la PSF.
- **Rango:** el del real ±1.5 µm más $2d\sin\beta$ hasta el foco del reflejo. Para d = 8 µm son ~11 µm más los márgenes: ~300 planos.
- **Fotones:** de 3 a 5 frames por plano para promediar. Beads brillantes (≥ 20 000 fotones por frame en foco), así los errores laterales quedan bajo 5 nm.
- **Barrido:** en una sola dirección (y repetido en la otra para medir histéresis), con una bead lejos del espejo como referencia de deriva.

## Cálculo del recorte de pupila

Espejo infinito, cono de colección arcsin(NA/n) = 55.6°. Un rayo del reflejo virtual llega al objetivo solo si cruza el plano del espejo ($w\cdot n > 0$).

| β (desde el eje) | luz perdida (pupila uniforme / aplanática) | corrimiento del centroide por µm de desenfoque |
|---|---|---|
| 40° | 8.2% / 10.0% | 95 / 118 nm |
| 45° | 4.7% / 5.9% | 57 / 73 nm |
| 50° | 1.8% / 2.4% | 24 / 32 nm |
| 56° (montaje anterior) | 0% | 0 |

Es óptica geométrica: el corrimiento es una cota del orden de magnitud a desenfoques de varios µm. A 45° el corrimiento entra 1:1 en la z triangulada ($\Delta m' = -\Delta z$), así que con |z'| de hasta 4.6 µm el sesgo puede llegar a ~300 nm si no se modela o calibra.
