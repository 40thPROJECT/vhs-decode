# vhs-decode-fast

A fork of [oyvindln/vhs-decode](https://github.com/oyvindln/vhs-decode) aimed at
one problem: **a full VHS tape takes a very long time to decode, and `--threads`
barely helps.**

Based on upstream commit `4a5479c2`. Everything upstream still works the same
way — the original documentation is kept as
[README_vhs-decode.md](README_vhs-decode.md).

**[English](#english) · [Español](#español)**

---

# English

## What this fork adds

### Faster decoding, same output

| change | gain | output |
|---|---|---|
| Rewind buffer no longer recopied on every read | **+16.9 %** | identical |
| Colour burst filtered in batches, not line by line | **+61 %** (without the Rust module) | bit-identical |
| Wow spline cached across `downscale()` calls | **+9 %** | bit-identical |
| `scipy.fft` instead of `numpy.fft` | ~23 % less time in FFTs | identical |
| Level check in one pass instead of two array scans | small | identical |
| TBC resampling spread across cores | varies | identical |

## New tools

### `decode_parallel.py` — several processes on one machine

```bash
python decode_parallel.py vhs --tape_format vhs --system ntsc --jobs 3 \
    capture.ldf output
```

Cuts the capture into one span per job, decodes them as independent processes and
stitches the results. Shows a live progress bar with frames done, current FPS and
time remaining.

Add `--no_merge` to leave the pieces as separate usable decodes, so each can be
exported and deleted before the next — merging a whole tape needs room for a
second copy of it.

**Be realistic about the gain.** On a 6-core i5-11400F this is worth about 10 %:

| | time | FPS | vs one process |
|---|---|---|---|
| one process, `-t 1` | 601 s | 2.99 | — |
| one process, `-t 6` | 636 s | 2.83 | 0.95x |
| `--jobs 3` | 554 s | 3.28 | **1.10x** |
| `--jobs 6` | 593 s | 3.06 | 1.02x |

With six jobs the CPU is pegged at 100 % while the disk is idle. Each decoder
uses ~1.6 cores even with `--threads 1`, so six jobs ask for ~9.6 cores on six
physical ones. The limit looks like memory bandwidth and shared L3, not code.
**On a machine with more cores this should scale much better** — but measure it,
do not assume.

### `split_capture.py` + `merge_tbc.py` — across several machines

```bash
# cut the capture into standalone pieces
python split_capture.py --parts 4 capture.ldf pieces/

# decode a piece on each machine, however you like
python decode.py vhs --tape_format vhs --system ntsc capture.part00.ldf out

# join the results, in tape order
python merge_tbc.py --manifest pieces/capture.parts.json --output tape \
    pc1.tbc pc2.tbc pc3.tbc pc4.tbc
```

Splitting is a **byte copy** — FLAC frames are self-contained, so a file made of
the original headers plus a run of whole frames is a valid `.ldf`. Nothing is
re-encoded and the pieces are bit-identical to the corresponding samples. Packed
and raw captures (`.lds`, `.s16`, …) split the same way.

Unlike parallel decoding on one machine, this scales nearly linearly — each
machine brings its own cores and memory bandwidth.

Keep the `.parts.json` manifest. Without it there is no way to know where each
decode belongs or how much overlap to trim.

## Requirements and caveats

* **Use a built checkout.** The compiled Cython extensions and the Rust module
  are worth far more than any of this; a fresh clone has neither. See
  [BUILD.md](BUILD.md).
* **Disk space.** A 2h45m NTSC tape is ~567 GB of `.tbc` (luma + chroma). Merging
  used to need a second full copy on top of that; it now frees each part as it is
  absorbed, but plan for the output plus one part.
* The seams between pieces are the trade-off: each job re-locks sync at its own
  starting point, so a frame at each join may differ slightly from a single
  continuous decode. Field parity is repaired across joins, so interlacing and
  chroma phase stay correct.

## How this was verified

Everything was measured on a real 2h45m NTSC capture (240 GB `.ldf`, 40 MSPS),
not on synthetic data. The bit-identical claims were checked byte for byte
against the unmodified decoder.

The full round trip — split a capture into three, decode each piece separately,
merge — gives **2694 fields against 2696** for a single continuous decode over
the same span, with the worst gap at a seam no worse than the worst gap the
decoder already produces inside a continuous decode.

## Detailed write-up

[docs/vhs-decode-fast/README.md](docs/vhs-decode-fast/README.md) — the reasoning
behind each change, what was tried and abandoned, and why `--threads` plateaus.

---

# Español

Un fork de [oyvindln/vhs-decode](https://github.com/oyvindln/vhs-decode)
dirigido a un solo problema: **decodificar una cinta VHS entera lleva muchísimo
tiempo, y `--threads` apenas ayuda.**

Basado en el commit `4a5479c2` del original. Todo lo del proyecto original sigue
funcionando igual — su documentación se conserva en
[README_vhs-decode.md](README_vhs-decode.md).

## Qué añade este fork

### Decodificación más rápida, misma salida

| cambio | ganancia | salida |
|---|---|---|
| El búfer de rebobinado ya no se recopia en cada lectura | **+16,9 %** | idéntica |
| Filtrado del burst de color por lotes, no línea a línea | **+61 %** (sin el módulo Rust) | bit a bit idéntica |
| Caché del spline de wow entre llamadas a `downscale()` | **+9 %** | bit a bit idéntica |
| `scipy.fft` en vez de `numpy.fft` | ~23 % menos tiempo en las FFT | idéntica |
| Comprobación de niveles en una pasada en vez de dos | pequeña | idéntica |
| Remuestreo del TBC repartido entre núcleos | variable | idéntica |

## Herramientas nuevas

### `decode_parallel.py` — varios procesos en una máquina

```bash
python decode_parallel.py vhs --tape_format vhs --system ntsc --jobs 3 \
    captura.ldf salida
```

Corta la captura en un tramo por trabajo, los decodifica como procesos
independientes y une los resultados. Muestra una barra de progreso en vivo con
frames hechos, FPS actuales y tiempo restante.

Añade `--no_merge` para dejar las piezas como decodes separados y utilizables,
así puedes exportar y borrar de una en una — unir una cinta entera necesita sitio
para una segunda copia de ella.

**Sé realista con la ganancia.** En un i5-11400F de 6 núcleos vale alrededor del
10 %:

| | tiempo | FPS | vs un proceso |
|---|---|---|---|
| un proceso, `-t 1` | 601 s | 2,99 | — |
| un proceso, `-t 6` | 636 s | 2,83 | 0,95x |
| `--jobs 3` | 554 s | 3,28 | **1,10x** |
| `--jobs 6` | 593 s | 3,06 | 1,02x |

Con seis trabajos la CPU está al 100 % mientras el disco está ocioso. Cada
decodificador usa ~1,6 núcleos incluso con `--threads 1`, así que seis trabajos
piden ~9,6 núcleos sobre seis físicos. El límite parece el ancho de banda de
memoria y la L3 compartida, no el código. **En una máquina con más núcleos esto
debería escalar mucho mejor** — pero mídelo, no lo des por hecho.

### `split_capture.py` + `merge_tbc.py` — entre varias máquinas

```bash
# cortar la captura en piezas autónomas
python split_capture.py --parts 4 captura.ldf piezas/

# decodificar una pieza en cada máquina, como prefieras
python decode.py vhs --tape_format vhs --system ntsc captura.part00.ldf salida

# unir los resultados, en orden de cinta
python merge_tbc.py --manifest piezas/captura.parts.json --output cinta \
    pc1.tbc pc2.tbc pc3.tbc pc4.tbc
```

Partir es una **copia de bytes** — los frames FLAC son autocontenidos, así que un
fichero formado por las cabeceras originales más una tirada de frames enteros es
un `.ldf` válido. No se recodifica nada y las piezas son idénticas bit a bit a
las muestras correspondientes. Las capturas empaquetadas y crudas (`.lds`,
`.s16`, …) se parten igual.

A diferencia del paralelismo en una sola máquina, esto escala casi linealmente:
cada máquina aporta sus propios núcleos y su propio ancho de banda de memoria.

Guarda el manifiesto `.parts.json`. Sin él no hay forma de saber dónde va cada
decode ni cuánto solape recortar.

## Requisitos y advertencias

* **Usa un checkout compilado.** Las extensiones Cython y el módulo Rust valen
  mucho más que todo esto junto; un clon recién bajado no trae ninguno. Ver
  [BUILD.md](BUILD.md).
* **Espacio en disco.** Una cinta NTSC de 2 h 45 son ~567 GB de `.tbc` (luma +
  croma). Antes la unión necesitaba además una segunda copia completa; ahora
  libera cada parte según la absorbe, pero cuenta con la salida más una parte.
* Las costuras entre piezas son el precio a pagar: cada trabajo vuelve a
  enganchar el sync en su propio punto de partida, así que un frame en cada unión
  puede diferir ligeramente de un decode continuo. La paridad de campo se repara
  en las uniones, así que el entrelazado y la fase de croma se mantienen
  correctos.

## Cómo se verificó

Todo está medido sobre una captura NTSC real de 2 h 45 (`.ldf` de 240 GB,
40 MSPS), no sobre datos sintéticos. Las afirmaciones de "bit a bit idéntica" se
comprobaron byte a byte contra el decodificador sin modificar.

El ciclo completo — partir una captura en tres, decodificar cada pieza por
separado, unir — da **2694 campos frente a 2696** de un decode continuo sobre el
mismo tramo, con el peor salto en una costura no peor que el peor salto que el
decodificador ya produce dentro de un decode continuo.

## Documentación detallada

[docs/vhs-decode-fast/LEEME.md](docs/vhs-decode-fast/LEEME.md) — el razonamiento
detrás de cada cambio, lo que se probó y se descartó, y por qué `--threads` se
estanca.

---

## Credits

All the hard work is [oyvindln/vhs-decode](https://github.com/oyvindln/vhs-decode)
and its contributors. This fork only makes it finish sooner.
