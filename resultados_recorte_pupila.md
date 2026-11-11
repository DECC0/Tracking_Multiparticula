# Recorte de pupila del reflejo: cuánto pesa y qué hace con la z

Modelo implementado en `simulador_nuevo` (`recorte_pupila_reflejo=True`, `PSFReflejo`): con un espejo plano infinito, el rayo de dirección w que parece salir del reflejo virtual llega al objetivo solo si cruza el plano del espejo hacia la muestra, w·n > 0, con n = cos β m̂ + sin β ẑ (+z hacia el objetivo, β desde el eje óptico). La PSF del reflejo se calcula directo para cada subposición (una FFT de la pupila recortada) y **no se renormaliza**: suma la fracción de la pupila que pasa, así que `ratio_reflejo` queda como la reflectividad del espejo. Todas las cifras salen de

```bash
$PY figuras/recorte_pupila.py --red <corrida>/evaluacion_<test> <corrida>/evaluacion_<test>_recorte [--red ...]   # ~100 s
```

que escribe `figuras/fig_recorte_pupila.png/.pdf` (resumen), `figuras/fig_recorte_pupila_psf.png/.pdf` (PSF completa y recortada) y `figuras/recorte_pupila_*.csv`. «lejos» y «cerca» son las dos variantes de la PSF esférica (`anillos_hacia_objetivo` False / True).

## Resumen

| | resultado |
|---|---|
| luz perdida (β = 40 / 45 / 50°) | 8.2 / 4.7 / 1.8 % (nada desde 55.6°) |
| corrimiento del centroide (rejilla completa, β = 45°) | geométrico 57 nm/µm de desenfoque sin máscara de fase; con la esférica, hasta ±400 nm, pero casi todo en el halo |
| sesgo de la z triangulada si se ajusta con la PSF completa | «cerca»: ≤ 3 nm (z' ≤ −1.5 µm), ≤ 9 nm en todo el rango; «lejos»: −15 a −32 nm con z' ≤ −2.5 µm, y depende de la ROI |
| CRLB conjunta de z | el recorte la empeora 0–14 % (d ≥ 0.5 µm) |
| red entrenada sin recorte, sobre datos con recorte | sesgo de z pareado de ±2 nm (+7 a +9 ± 3 nm a d = 3–4 µm); recall −2 a −3 puntos a d < 1 µm; RMSE z sin cambio |
| ratio calibrado (fotometría de los datos reales, «cerca») | 0.55–1.0 con β en [40, 50] y z del real en ±1.5 µm (0.63–1.0 a 45°); no se puede afinar más sin conocer la z del real |
| rango recomendado | `ratio_reflejo_rango: [0.5, 1.0]` con `recorte_pupila_reflejo: true` (ya en `espejo.yaml` y `celula.yaml`) |
| variante de la PSF | los datos reales favorecen «cerca»: con «lejos» un reflejo a z' ≈ −4.5 µm sería un anillo de ~8 µm |

## 1. Luz perdida contra β

| β | onda (simulador) | geométrica, pupila uniforme | geométrica, aplanática |
|---|---|---|---|
| 0° | 50.00 % | 50.00 % | 50.00 % |
| 20° | 27.18 % | 27.18 % | 29.14 % |
| 30° | 16.95 % | 16.96 % | 19.17 % |
| 40° | 8.24 % | 8.24 % | 9.96 % |
| 45° | 4.70 % | 4.67 % | 5.87 % |
| 50° | 1.81 % | 1.82 % | 2.38 % |
| 55° | 0.13 % | 0.07 % | 0.09 % |

- En el cálculo de onda, la pérdida es exactamente la integral de la máscara sobre la pupila (Parseval), así que coincide con la geométrica de pupila uniforme salvo por la discretización de la rejilla.
- La pupila del simulador tiene amplitud uniforme. Con la apodización aplanática (1/cos θ en intensidad) se pierde ~25 % más (5.9 % a 45°), porque pesan más los rayos marginales, que son los que se recortan.
- Dentro del rango de entrenamiento (β 40–50°) la pérdida va de 8.2 % a 1.8 %.

## 2. Forma y corrimiento del centroide contra z'

β = 45°, m̂ = +x; el espejo recorta el lado −m (`fig_recorte_pupila_psf`).

**Sin máscara de fase («ninguna»)**, el centroide de la PSF completa se corre exactamente lo que da la óptica geométrica: +57 nm por µm de desenfoque hacia +m con z' < 0, y hacia −m con z' > 0. Con las dos variantes de anillos da lo mismo, como debe ser (`test_recorte.py`).

**Con la esférica**, la aberración esférica pone el foco de los rayos marginales (los que se recortan) a ~±2.5 µm del foco paraxial. Por eso el corrimiento del centroide de la rejilla completa ya no se anula en z' = 0:
- «lejos»: de −143 nm a z' = 0 a +112 nm a z' = −4.5 µm;
- «cerca»: de +143 nm a +377 nm a z' = −4.5 µm.

Ese corrimiento está casi todo en el halo, no en el núcleo:
- Con el centroide iterativo de r = 1 µm (el de los datos reales), «cerca» se corre menos de 20 nm en todo z' ≤ −1 µm.
- En «lejos» el centroide del núcleo es errático (−270 a +320 nm), porque a z' ≲ −2 µm la PSF es un sistema de anillos sin núcleo claro.

Forma:
- La diferencia L2 relativa con la PSF completa va de 6 a 19 %.
- La asimetría del perfil a lo largo de m es chica (skewness ≤ 0.07).
- «lejos» se angosta a lo largo de m entre z' = −2.5 y −4.5 µm (anchos m/t 0.90–0.92). En «cerca» el núcleo casi no cambia (anchos 0.97–1.01, y la misma luz dentro de 1.5 µm).

## 3. Sesgo de la z triangulada

Se ajusta por MLE de Poisson (x, y, z', N y fondo libres) la imagen sin ruido del reflejo recortado (5000 fotones, fondo 20) con el modelo de PSF completa, como haría un localizador que no conoce el recorte. Así sale el sesgo asintótico.

La z triangulada depende solo de la coordenada m del reflejo (su coordenada t es la del real), así que su sesgo es exactamente dz = −dm / sin 2β, es decir −dm a 45°. El control con la PSF completa da sesgo 0.

| z' [µm] | −5.5 | −4.5 | −3.5 | −2.5 | −1.5 | −0.5 | +0.5 |
|---|---|---|---|---|---|---|---|
| «lejos», β 45, ROI 31 px | −32 | −32 | −20 | −15 | +4 | +7 | 0 |
| «lejos», β 45, ROI 61 px | −19 | +7 | +12 | −10 | +4 | +7 | 0 |
| «lejos», β 40 / 50, ROI 31 | −28 / −46 | −26 / −25 | −24 / −9 | −27 / −5 | −1 / +5 | +9 / +2 | +7 / −1 |
| «cerca», β 45, ROI 31 y 61 px | −2 | 0 | 0 | +1 | +3 | 0 | −7 |
| «cerca», β 40 / 50, ROI 31 | −3 / −1 | +1 / 0 | −2 / 0 | +4 / +1 | +2 / +2 | −7 / +1 | −9 / −2 |

- El sesgo de z' del propio ajuste es de ±40 nm como mucho (`recorte_pupila_sesgo.csv`).
- La cota geométrica de `resultados_calibracion_espejo.md` (hasta ~300 nm a |z'| = 4.6 µm) sobreestima mucho: el ajuste sigue el núcleo de la PSF, y en el núcleo el recorte casi no mueve nada.
- Con «cerca» el sesgo es despreciable frente a la precisión (CRLB de 10–30 nm).
- Con «lejos», a z' ≤ −2.5 µm es de 15–45 nm, del orden de la ganancia del reflejo, y cambia con la ROI del ajuste: depende de cuánto halo de anillos entra.

## 4. CRLB conjunta

Es el modelo conjunto de `figuras/crlb_espejo.py` (real + reflejo con su posición atada a la del real, β = 45°, ratio 0.8, fondo 20), con el reflejo completo o recortado; en los dos casos el modelo es el correcto. Valores de √CRLB z en nm, RMS sobre z ∈ {−800, 0, 800}, para 5000 fotones:

| d [µm] | 0 | 0.5 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|
| solo | 34 | 34 | 34 | 33 | 33 | 33 |
| «lejos», completo / recortado | 102 / 51 | 9 / 10 | 9 / 10 | 14 / 16 | 21 / 22 | 25 / 26 |
| «cerca», completo / recortado | 102 / 51 | 9 / 9 | 9 / 9 | 9 / 10 | 18 / 18 | 29 / 29 |

- Con d ≥ 0.5 µm el recorte cuesta 0–14 % (con 2000 y 20000 fotones también).
- En d = 0 real y reflejo coinciden y el modelo conjunto es casi degenerado; la asimetría del recorte la rompe en parte.
- La ganancia del reflejo (×2–4 frente al emisor solo) se mantiene.

## 5. Red entrenada sin recorte sobre datos con recorte

Corridas piloto `piloto_dz100` y `piloto_dz50` (`espejo.yaml` a 64 px, ±800 nm, «lejos», ratio 0.8, 10 épocas × 10000), evaluadas sobre el test `test_piloto_dz_b45_lejos` (1000 imágenes, seed 3) y sobre la misma muestra regenerada con recorte (`localizacion/datos/test_piloto_dz_b45_lejos_recorte`). Las dos tienen el mismo GT y las mismas posiciones. La regeneración sin recorte reproduce el test original bit a bit. El ruido de cámara sí difiere entre las dos, así que la comparación pareada por emisor elimina la configuración pero no el ruido. La diferencia media es el sesgo:

| dz100 (dz50) | recall sin → con | RMSE z sin → con [nm] | dz con − sin [nm] |
|---|---|---|---|
| todos | 0.787 → 0.780 (0.749 → 0.748) | 65.2 → 65.3 (63.7 → 65.3) | −0.7 ± 1.1 (+0.9 ± 1.1) |
| reflejo en el FOV, d 0–1 µm | 0.765 → 0.733 (0.737 → 0.719) | 64.2 → 63.9 (62.8 → 64.0) | −0.4 ± 2.8 (+2.0 ± 2.8) |
| reflejo en el FOV, d 1–2 µm | 0.795 → 0.790 (0.769 → 0.764) | 64.4 → 67.8 (62.2 → 68.7) | −1.5 ± 2.8 (−0.9 ± 3.0) |
| reflejo en el FOV, d 2–3 µm | 0.832 → 0.823 (0.803 → 0.800) | 63.7 → 65.4 (61.0 → 64.8) | −1.7 ± 2.6 (+0.9 ± 2.7) |
| reflejo en el FOV, d 3–4 µm | 0.804 → 0.811 (0.751 → 0.755) | 65.0 → 64.4 (58.0 → 58.7) | +7.4 ± 3.1 (+8.5 ± 3.1) |

- La red casi no cambia. El único efecto claro es la pérdida de recall cerca del espejo (−2 a −3 puntos a d < 1 µm, donde las dos PSF se superponen) y un sesgo de +8 nm a d = 3–4 µm.
- Es coherente con que estas redes usan poco el reflejo para la z (`resultados_crlb_espejo.md`).
- Una red que sí lo explote heredaría el sesgo de la sección 3. Por eso conviene entrenar con el recorte, que es lo que ahora hacen `espejo.yaml` y `celula.yaml`.

## 6. Calibración de `ratio_reflejo` por fotometría

Se aplica la misma fotometría que a los datos reales (`resultados_datos_reales.md`, sección 3):
- disco de r = 1.5 µm;
- fondo = mediana del anillo [r+3, r+10] px sin el otro punto;
- centros por centroide iterativo.

Se usa sobre imágenes simuladas sin ruido con `ratio_reflejo` = 1 y recorte, la separación medida (4.5 µm) y β 40/45/50°. Así sale Q1 = cociente medido por unidad de ratio; la fotometría es lineal, de modo que ratio = Q_medido / Q1.

**Q1 depende mucho de la z del real**, que no se conoce. Con «cerca» y β = 45°:

| z del real [µm] | −2.5 | −2.0 | −1.5 | −1.0 | −0.5 | 0 | +0.5 | +1.0 | +1.5 | +2.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | 0.34 | 0.35 | 0.36 | 0.38 | 0.41 | 0.44 | 0.42 | 0.46 | 0.58 | 0.91 |

- Con y sin recorte, Q1 es casi igual: la luz recortada sale del halo, no de la apertura.
- **Con «lejos», Q1 es negativo o casi nulo** para casi toda z: el reflejo a z' ≈ −4.5 µm es un anillo ancho sin núcleo, y el fondo del anillo de la fotometría queda por encima del disco.
- Además, el ajuste de la curva Q(r) medida (r = 0.5 a 2.2 µm) deja con «lejos» residuos 3 a 4 veces mayores que con «cerca» (salvo a β = 50°).
- Los datos (un reflejo compacto, con Q ≈ 0.43 positivo) **descartan «lejos»** con esta PSF. Es la misma conclusión del análisis del frame real, ahora con una cifra.
- El ajuste de la forma de Q(r) tampoco fija la z del real: residuos de ~0.1 en un rango amplio de z, y estimadores que dan ρ de 0.5 a 1.3 según el canal.

Intervalo calibrado («cerca», z del real en ±1.5 µm, ratio ≤ 1 por ser una reflectividad):

| β | Q1 | canal derecho (Q 0.42–0.46) | canal izquierdo (Q 0.37–0.44) |
|---|---|---|---|
| 40° | 0.43–0.67 | 0.63–1.0 | 0.55–1.0 |
| 45° | 0.36–0.58 | 0.72–1.0 | 0.63–1.0 |
| 50° | 0.28–0.48 | 0.88–1.0 | 0.78–1.0 |

**Rango fijado: `ratio_reflejo_rango: [0.5, 1.0]`**, en `espejo.yaml` y `celula.yaml`, con `recorte_pupila_reflejo: true`. Cubre el intervalo calibrado con todas las incógnitas (0.55–1.0) más un margen hacia abajo. Un espejo metálico o dieléctrico real está en 0.9–0.99, así que la parte alta del rango es la físicamente esperable.

Salvedades:
- Es una sola partícula, del montaje anterior.
- La máscara de esos datos («sphericalHalfIIntensity») puede no ser la Z4⁰ de amplitud 1.5π del simulador.
- El registro entre canales no está calibrado.

Para fijar el ratio de verdad hace falta el z-stack de beads de `resultados_calibracion_espejo.md` (punto 4): da cada bead en su foco, así que la z del real deja de ser incógnita.

## Supuestos y lo que falta

- **Espejo infinito.** Un borde finito recorta más el reflejo, cuanto más lejos esté la muestra del espejo: la propuesta de calibración ya prevé medir la dependencia del cociente con d.
- **El real no se recorta.** Es correcto si el espejo no sube por encima del plano focal más que unos pocos µm. Los rayos directos del real con w·n < 0 chocan con el espejo por encima del emisor. Con óptica geométrica, a 45° el real pierde:
  - 0 % si el borde superior del espejo está a H ≤ 2 µm sobre el plano focal (con d ≥ 1 µm);
  - 0.9 % con H = 5 µm y d = 1 µm, y 2.4 % con d = 0.5 µm;
  - 3.5 % con H = 20 µm;
  - 4.7 % (lo mismo que el reflejo) con H → ∞.

  Si el espejo nace en el cubreobjetos, a pocos µm del plano focal, la aproximación es buena; si no, habría que recortar también al real.
- La pupila tiene amplitud uniforme (sin apodización aplanática): la pérdida real sería ~25 % mayor que la simulada.
- Óptica escalar, sin polarización.
- Costo: ~2 ms por PSF del reflejo. Una muestra de `celula.yaml` pasa de 2.4 a 13 ms con 8 workers (de 11 a 37 ms en un proceso), lejos de los ~53 ms de GPU por muestra en la RTX 3060. En Hypatia conviene `CPUS=8` (`hypatia/README.md`).
