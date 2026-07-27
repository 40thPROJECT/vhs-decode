# Decodificación más rápida

Cambios dirigidos a un solo problema: decodificar una cinta VHS entera lleva
muchísimo tiempo, y `--threads` apenas ayuda.

Todo lo de aquí está medido sobre una captura NTSC real de 2 h 45 min (`.ldf` de
240 GB, 40 MSPS) en un Intel i5-11400F — 6 núcleos, 12 hilos. Los números en otro
hardware serán distintos; el razonamiento no debería.

* [Por qué `--threads` se estanca](#por-qué---threads-se-estanca)
* [Mejoras en un solo proceso](#mejoras-en-un-solo-proceso)
* [Búsqueda dentro de un .ldf](#búsqueda-dentro-de-un-ldf)
* [La trampa de la duración del .ldf](#la-trampa-de-la-duración-del-ldf)
* [Decodificación paralela en una máquina](#decodificación-paralela-en-una-máquina)
* [Repartir una captura entre varias máquinas](#repartir-una-captura-entre-varias-máquinas)
* [Herramientas añadidas](#herramientas-añadidas)
* [Lo que no funcionó](#lo-que-no-funcionó)

---

## Por qué `--threads` se estanca

`--threads` solo paraleliza la demodulación. La detección de sync, el cálculo de
las posiciones de línea y la corrección de base de tiempos van en un único hilo,
en orden, campo a campo, y ese hilo marca el techo.

Medido sobre la captura anterior, decodificando los mismos 1798 frames:

| | tiempo | FPS |
|---|---|---|
| `--threads 1` | 601 s | 2,99 |
| `--threads 6` | 636 s | 2,83 |

Más hilos es ligeramente *peor*. Pasados unos dos hilos, los trabajadores extra
se pasan el tiempo esperando a la etapa serie y peleando por el ancho de banda de
memoria.

## Mejoras en un solo proceso

Todas producen una salida bit a bit idéntica a la del decodificador original,
salvo donde se indique.

### El búfer de rebobinado ya no se copia en cada lectura — `lddecode/utils.py`

`LoadLDF._read_data` y `LoadFFmpeg._read_data` mantienen una ventana de
rebobinado (2 MB y 16 MB) para poder retroceder un poco. Ambos añadían los datos
nuevos y volvían a recortar el búfer entero:

```python
self.rewind_buf += data
self.rewind_buf = self.rewind_buf[-self.rewind_size:]
```

Eso copia la ventana completa en cada lectura — varios GB de `memcpy` por campo,
para tirar unos pocos kB del principio. Ahora el búfer es un `bytearray` que solo
se recorta cuando ha crecido más del doble de la ventana.

**Medido: 16,9 % más rápido** (137,0 s → 113,8 s para 300 frames, dos
repeticiones de cada uno, intercaladas). Es la mayor ganancia de todo esto, y son
cuatro líneas.

### Filtrado del burst de color por lotes — `vhsdecode/chroma.py`

`_get_upconverted_burst` llamaba a `sosfiltfilt` una vez por línea — unas 800
llamadas por campo, cada una sobre un trozo de ~50 muestras. Cada llamada
recalculaba desde cero las condiciones iniciales del filtro, que cuesta mucho más
que filtrar las muestras.

`_prefilter_burst_windows` ahora filtra la ventana del burst de todas las líneas
para cada fase del heterodino en una sola llamada por fase — 4 por campo en vez
de ~800. `sosfiltfilt` rellena y filtra cada fila de un array 2-D de forma
independiente, así que las filas son idénticas a lo que producían las llamadas
línea a línea.

**Medido: +61 %** sin el módulo Rust, prácticamente neutro con él. Salida bit a
bit idéntica.

### Caché del spline de wow — `lddecode/core.py`

`downscale()` corre dos o tres veces por campo (luma, burst de croma, croma) y
cada llamada reajustaba y reevaluaba el mismo spline de interpolación.
`computewow_scaled` ahora cachea el resultado, indexado por las posiciones de
línea y la geometría del campo — las posiciones se refinan a mitad de campo, así
que la caché compara contenidos en vez de cachear una sola vez.

**Medido: +9 %.** Salida bit a bit idéntica.

### Remuestreo del TBC repartido entre núcleos — `lddecode/utils.py`

El bucle de remuestreo sinc-enventanado de `scale_field` se separa en
`_scale_field_resample` y corre sobre `prange`. Cada muestra de salida lee una
ventana fija de la entrada y escribe una posición de la salida, así que las
iteraciones son independientes. El preámbulo sigue siendo secuencial: el filtro
de suavizado del wow es una recurrencia y no se puede paralelizar.

### Descartados al rebasar sobre el original actual

Tres cambios que llevaba este fork han desaparecido, porque el proyecto original
llegó al mismo sitio: `check_levels` en una pasada, un fallback en Python para el
módulo Rust, y una guarda sobre la división por cero de `_sync_to_burst`. Tenían
sentido contra la base del 16 de julio y ahora son redundantes. El cambio de
`numpy.fft` a `scipy.fft` iba en el mismo commit que `check_levels` y se fue con
él; el original sigue usando `numpy.fft` en `process.py`, así que ese conviene
recuperarlo.

## Búsqueda dentro de un .ldf

**Este es el cambio que hace posible todo lo demás sobre una captura
comprimida.**

Los `.ldf` de `ld-compress` no llevan tabla de seek. En una captura lo bastante
larga como para que el contador de muestras de 36 bits de FLAC no la pueda
describir, ffmpeg recurre a estimar las posiciones de byte a partir del bitrate.
En la captura de 2 h 45 esa estimación se quedaba **25 veces corta**: pedir la
muestra a un tercio del fichero aterrizaba cerca del principio.

`LoadLDF` devolvía muestras correctas igualmente — decodifica hacia delante y
descarta todo lo anterior al objetivo — pero a unos 80 M muestras/s eso significa
que el último trabajo de un decode paralelo de 6 tardaría **66 minutos** solo en
llegar a su punto de partida. En la práctica, `.ldf` no tenía acceso aleatorio en
absoluto: cada búsqueda era un escaneo lineal desde el principio, invisible en un
fichero pequeño y fatal en uno grande.

`lddecode/flacseek.py` lo resuelve. Los frames FLAC llevan su propia posición:
en un stream de blocksize fijo — que es lo que escribe `ld-compress` — la primera
muestra del frame N es `N * blocksize`. Así que el fichero es navegable sin
índice:

1. Se biseca el fichero por posición de byte, resincronizando en cabeceras de
   frame. Los candidatos se validan por palabra de sync, CRC-8, coherencia con
   STREAMINFO, y exigiendo que el frame *siguiente* continúe la secuencia — una
   cabecera aislada que parece válida puede aparecer por azar dentro del audio
   comprimido.
2. Se le entrega al decodificador un stream spliceado: las cabeceras originales
   seguidas de los bytes del fichero desde ese offset. ffmpeg ve un FLAC
   perfectamente formado que resulta que empieza a mitad de la grabación.

La bisección es de falsa posición con una bisección simple forzada cada dos
pasos. La interpolación sola se atasca: cuando la tasa estimada es muy próxima a
la real, la conjetura cae un byte por debajo del límite, encuentra el mismo frame
otra vez y el intervalo se encoge un byte por iteración. Eso costaba 45.568
escaneos en un fichero de 107 MB. Con la salvaguarda son 5.

Medido sobre la captura de 240 GB:

| muestra pedida | aterriza en | error | tiempo |
|---|---|---|---|
| 39.543.780.000 | 39.543.779.328 | 672 muestras | 0,00 s |
| 197.718.900.000 | 197.718.898.688 | 1.312 | 0,00 s |
| 355.894.020.000 | 355.894.018.048 | 1.952 | 0,02 s |

Siempre dentro de un bloque, prácticamente instantáneo. Los ficheros que no se
pueden posicionar así — envueltos en ogg o de blocksize variable — recurren al
seek del contenedor.

Las posiciones son relativas al primer frame del fichero, así que un trozo
recortado de una captura más larga también funciona: sus frames conservan la
numeración que tenían en el original.

## La trampa de la duración del .ldf

`decode_parallel.py` necesita saber cuánto dura una captura para poder partirla.
La fuente evidente miente.

En la captura de 2 h 45, el contenedor declaraba **272.629.760 muestras (6,8 s)**
frente a las **395.437.800.000 reales (164,8 min)** — un factor de ~1450. El
valor sale directamente de STREAMINFO, donde el codificador escribió un recuento
equivocado; una captura de más de 2³⁶ muestras no se puede describir ahí.

Fiarse de él decodificaba el 0,07 % de la cinta e informaba de éxito. Ese es el
peor tipo de fallo: silencioso, y solo detectable si te fijas en que la salida es
muchísimo más corta de lo que debería.

Dos defensas:

* **Contrastar con el tamaño del fichero.** El audio comprimido nunca ocupa más
  que las muestras que codifica, así que declarar 545 MB de muestras para un
  fichero de 240 GB no es una declaración utilizable.
* **Leer el sidecar de la herramienta de captura.** El DomesDay Duplicator y el
  MISRC escriben `<captura>.json` junto a la captura con
  `captureInfo.durationInMilliseconds`, que sí es fiable.

Si fallan ambas, las herramientas se niegan y piden `--total_samples` o
`--duration` en vez de adivinar.

## Decodificación paralela en una máquina

`decode_parallel.py` corta la captura en un tramo contiguo por trabajo,
decodifica los tramos como procesos independientes y vuelve a unir los
resultados.

En esta máquina de 6 núcleos vale alrededor de un 10 %:

| | tiempo | FPS | vs un proceso |
|---|---|---|---|
| un proceso, `-t 1` | 601 s | 2,99 | — |
| un proceso, `-t 6` | 636 s | 2,83 | 0,95x |
| `--jobs 3` | 554 s | 3,28 | **1,10x** |
| `--jobs 6` | 593 s | 3,06 | 1,02x |

Con seis trabajos corriendo, la CPU está clavada al 100 % mientras el disco está
casi ocioso, y cada decodificador usa ~1,6 núcleos pese a `--threads 1` (el
`parallel=True` de numba y las FFT con hilos). Seis trabajos piden ~9,6 núcleos
sobre seis físicos. El límite parece ser el ancho de banda de memoria y la L3
compartida, más que algo arreglable en código: seis copias de un espacio de
trabajo grande de FFT no caben en 12 MB.

**En una máquina con más núcleos y más ancho de banda esto debería escalar mucho
mejor.** No tomes ese 1,10x como una propiedad del método; es una propiedad de
esta CPU. Repartir entre máquinas (más abajo) esquiva el problema por completo.

### Las costuras

Cada trabajo vuelve a enganchar el sync en su propio punto de partida, así que
las uniones requieren cuidado:

* Los trabajos se pasan de su tramo a propósito; la unión recorta el solape
  usando `fileLoc`, cortando cada trabajo donde empezó realmente el siguiente y
  no en el punto de corte nominal. Un decodificador al que se le pide saltar a la
  muestra N puede engancharse a un campo que empieza algo antes de N, y cortar en
  N lo eliminaría de ambos lados.
* La paridad de campo se repara en cada unión. Sin eso, todos los frames
  posteriores emparejan los dos campos equivocados.
* Un trabajo que no produjo campos se salta, para que el sobrante del anterior no
  duplique el contenido del trabajo siguiente.
* Las partes se copian campo a campo y se liberan según se absorben. Antes la
  unión mantenía en disco una segunda copia completa del decode — 1.135 GB de
  pico para esta cinta, que habría fallado justo al final tras 22 horas
  decodificando. Ahora el pico es la salida más una parte.

Verificado sobre salida real: `fileLoc` estrictamente creciente, sin posiciones
duplicadas, paridad alternando, luma y croma exactos al byte contra el número de
campos, y el mayor salto en una costura no peor que el mayor salto que el
decodificador ya produce dentro de un decode continuo.

### `--no_merge`

Deja las piezas como decodes separados y utilizables por separado en vez de
unirlos, para cuando no hay sitio para una copia unida de una cinta entera. El
sobrante se recorta en sitio truncando los ficheros — no se reescribe ningún dato
ni hace falta espacio extra — así que las piezas siguen encajando exactamente.
Concatenarlas da un resultado idéntico byte a byte al de unirlas.

## Repartir una captura entre varias máquinas

`split_capture.py` corta la propia captura en ficheros autónomos.

Como los frames FLAC son autocontenidos, un fichero formado por las cabeceras
originales más una tirada de frames enteros es un `.ldf` válido que decodifica
ese tramo de cinta. Por tanto partir es una **copia de bytes** — no se recodifica
nada y las piezas son idénticas bit a bit a las muestras correspondientes del
original. Las capturas empaquetadas y crudas (`.lds`, `.s16`, `.r8`, …) se parten
igual, en fronteras de grupo de muestras.

Cada pieza lleva su propio recuento de muestras corregido en STREAMINFO, y se
pone a cero el MD5 del stream, porque el del original nunca puede cuadrar con una
pieza.

Las piezas se solapan un poco (2 s por defecto) porque un decodificador necesita
un momento para enganchar el sync al arrancar en frío. El solape queda registrado
en el manifiesto y lo recorta `merge_tbc.py`.

Ciclo completo verificado sobre RF real: partir una captura de 45 s en tres,
decodificar cada pieza por separado, unir, y comparar contra decodificar el
conjunto de una sola pasada:

| | |
|---|---|
| campos, unido vs una pasada | **2694 vs 2696** |
| tramo cubierto | 0,01–44,97 s en ambos |
| peor salto en una costura | idéntico al peor salto interno de la pasada única |

La diferencia de dos campos es la reparación de paridad en las dos uniones.

A diferencia del paralelismo en una sola máquina, esto escala casi linealmente:
cada máquina aporta sus propios núcleos y su propio ancho de banda de memoria.

## Herramientas añadidas

```bash
# Decodificar con varios procesos en una máquina
python decode_parallel.py vhs --tape_format vhs --system ntsc --jobs 3 \
    captura.ldf salida

# ... o dejar las piezas separadas, para exportar y borrar de una en una
python decode_parallel.py vhs --tape_format vhs --system ntsc --jobs 3 \
    --no_merge captura.ldf salida

# Cortar una captura en piezas autónomas para otras máquinas
python split_capture.py --parts 4 captura.ldf piezas/

# Unir los .tbc resultantes, en orden de cinta
python merge_tbc.py --manifest piezas/captura.parts.json --output cinta \
    pc1.tbc pc2.tbc pc3.tbc pc4.tbc
```

`decode_parallel.py` muestra una barra de progreso en vivo con los frames hechos,
los FPS actuales y el tiempo restante estimado. Lee el progreso del tamaño de los
ficheros `.tbc` y no de la salida de los decodificadores, que va en bloques y se
retrasa más de cien frames.

Guarda el manifiesto `.parts.json` — sin él no hay forma de saber dónde va cada
decode ni cuánto solape recortar.

## Cuando un trabajo se para antes de tiempo

Un decodificador que se topa con un error no controlado lo imprime, guarda lo que
lleva y **sale con codigo 0**.  Desde fuera parece que termino normalmente, asi
que el driver informo de "job 1 finished" y siguio adelante.

En una cinta real de 2 h 45 eso ocurrio a los 21 minutos de un tramo de 55, y
**se perdieron 34 minutos de cinta** de en medio del resultado.  La unica senal
fue un aviso de hueco durante la union, horas despues.

El fallo era una division por cero en el sincronismo de burst NTSC del original,
`vhsdecode/field.py:_sync_to_burst`:

```python
scale = burst_center_distance / (outlinelen * (burst_center_distance / line_length))
```

`line_length` vale cero cuando dos posiciones de linea consecutivas coinciden - un
campo degenerado, que una cinta ruidosa acaba produciendo.  La excepcion sale de
`Field.process()` y termina la decodificacion.

Las dos divisiones se cancelan: la expresion es `line_length / outlinelen`.  El
proyecto original llego a la misma conclusion por su cuenta y reescribio la
funcion - su codigo actual es `scale = line_length * inv_outlinelen`, sin ninguna
division que pueda fallar - asi que este fork no lleva arreglo propio para eso.
Lo que elimina el crash es rebasar sobre su trabajo.

Aparte, `decode_parallel.py` compara ahora lo que cubrio cada trabajo con lo que
se le pidio y lo dice en cuanto terminan, en vez de dejar que aparezca al unir o
que no aparezca.

Merece la pena dejarlo escrito: un decode de un solo proceso se habria parado en
ese mismo campo y habria perdido todo lo posterior - 88 minutos en vez de 34.
Repartir el trabajo contuvo el dano, y eso es una ventaja del decode paralelo que
no tiene nada que ver con la velocidad.

## Rellenar un hueco despues

`insert_tbc.py` mete un nuevo decode del tramo que falta dentro del hueco:

```bash
python insert_tbc.py --into cinta.tbc --insert hueco.tbc
```

La pieza va *dentro* del fichero, y eso `merge_tbc.py` no lo sabe hacer: une
piezas una detras de otra.  Partir el fichero y volver a unir necesitaria sitio
para una segunda copia del decode entero, cientos de gigabytes en una cinta
completa.

En su lugar se alarga el fichero por el tamano del trozo insertado y se desplaza
la cola, hacia atras desde su final para que nada se sobrescriba antes de haberse
copiado.  El unico espacio extra necesario es el del propio trozo.  El sobrante
de ambos lados se recorta por `fileLoc` y la paridad de campo se repara en las
dos uniones nuevas.

Los metadatos se escriben al final: hasta entonces el fichero sigue cuadrando con
su indice anterior, asi que una ejecucion interrumpida es recuperable.
`--dry-run` dice lo que haria sin tocar nada.

## Lo que no funcionó

**CUDA.** Las FFT por lotes en una RTX 3090 van unas 15 veces más rápido que en
CPU, pero que eso ayude depende de dónde se va realmente el tiempo, y eso cambia
según la captura. Sobre una captura sintética el demodulador no era el cuello de
botella y la ganancia proyectada era del 10-25 %. Sobre la captura real,
`demodblock` es el 70-80 % del tiempo y son sobre todo FFT, lo que hace mucho más
atractivo un camino por GPU. No se ha intentado aquí; perfila tu propio material
antes de dar por buena cualquiera de las dos conclusiones.

**Más hilos.** Ver el principio de este documento.
