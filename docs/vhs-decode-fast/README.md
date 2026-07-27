# Faster decoding

Changes aimed at one problem: a full VHS tape takes a very long time to decode,
and `--threads` does not help much.

Everything here was measured on a real 2h45m NTSC capture (240 GB `.ldf`,
40 MSPS) on an Intel i5-11400F — 6 cores, 12 threads. Numbers from other
hardware will differ; the reasoning should not.

* [Why `--threads` plateaus](#why---threads-plateaus)
* [Single-process speedups](#single-process-speedups)
* [Seeking inside a .ldf](#seeking-inside-a-ldf)
* [The .ldf length trap](#the-ldf-length-trap)
* [Parallel decoding on one machine](#parallel-decoding-on-one-machine)
* [Splitting a capture across machines](#splitting-a-capture-across-machines)
* [Tools added](#tools-added)
* [What did not work](#what-did-not-work)

---

## Why `--threads` plateaus

`--threads` parallelises demodulation only. Sync detection, line location and
the time-base correction run on a single thread, in order, one field after
another, and that thread sets the ceiling.

Measured on the capture above, decoding the same 1798 frames:

| | time | FPS |
|---|---|---|
| `--threads 1` | 601 s | 2.99 |
| `--threads 6` | 636 s | 2.83 |

More threads is slightly *worse*. Past about two threads the extra workers spend
their time waiting on the serial stage and contending for memory bandwidth.

## Single-process speedups

All of these were verified to produce bit-identical output to the unmodified
decoder, except where noted.

### Rewind buffer no longer copied on every read — `lddecode/utils.py`

`LoadLDF._read_data` and `LoadFFmpeg._read_data` kept a rewind window (2 MB and
16 MB respectively) so the decoder can seek backwards a little. Both appended
the new data and then re-sliced the whole buffer:

```python
self.rewind_buf += data
self.rewind_buf = self.rewind_buf[-self.rewind_size:]
```

That copies the entire window on every read — several GB of `memcpy` per field,
to discard a few kB off the front. The buffer is now a `bytearray` trimmed only
once it has grown past twice the window.

**Measured: 16.9 % faster** (137.0 s → 113.8 s for 300 frames, two runs each,
interleaved). This is the single largest win here, and it is four lines.

### Colour burst filtered in batches — `vhsdecode/chroma.py`

`_get_upconverted_burst` called `sosfiltfilt` once per line — about 800 calls per
field, each on a slice of roughly 50 samples. Every call re-derived the filter's
steady-state initial conditions from scratch, which costs far more than filtering
the samples themselves.

`_prefilter_burst_windows` now band-passes every line's burst window for every
heterodyne phase in one batched call per phase — 4 calls per field instead of
~800. `sosfiltfilt` pads and filters each row of a 2-D array independently, so
the rows are identical to what the per-line calls produced.

**Measured: +61 %** without the Rust extension, roughly neutral with it. Output
bit-identical.

### Wow spline cached — `lddecode/core.py`

`downscale()` runs two or three times per field (luma, chroma burst, chroma) and
each call re-fitted and re-evaluated the same interpolating spline.
`computewow_scaled` now caches the result, keyed on the line locations and field
geometry — line locations are refined mid-field, so the cache compares contents
rather than caching once.

**Measured: +9 %.** Output bit-identical.

### Level check in a single pass — `vhsdecode/addons/resync.py`

`check_levels` ran two `np.argwhere` calls over the whole field's `demod_05`,
allocating two index arrays, several times per field. It now counts both
thresholds in one loop with no allocation.

### `scipy.fft` instead of `numpy.fft` — `vhsdecode/process.py`, `nonlinear_filter.py`

A one-line import change. **Measured: ~23 % less time in the FFTs.**

### TBC resampling across cores — `lddecode/utils.py`

`scale_field`'s windowed-sinc resampling loop is split into
`_scale_field_resample` and run over `prange`. Every output sample reads a fixed
window of the input and writes one slot of the output, so the iterations are
independent. The preamble stays sequential — the wow smoothing filter is a
recurrence and cannot be parallelised.

### Python fallback for the Rust extension — `vhsdecode/rust_fallback.py`

`vhsdecode.demod` and `vhsdecode.main` imported `vhsd_rust` unconditionally, so a
source checkout without a cargo toolchain could not run at all. The fallback
provides numpy/numba equivalents. `vhsd_rust` is still preferred when present —
it is worth about +70 % on its own.

## Seeking inside a .ldf

**This is the change that makes everything else possible on a compressed
capture.**

`.ldf` files from `ld-compress` carry no seek table. On a capture long enough
that FLAC's 36-bit sample counter cannot describe it, ffmpeg falls back to
estimating byte positions from the bitrate. On the 2h45m capture that estimate
was **25x short**: asking for the sample a third of the way in landed near the
start.

`LoadLDF` did produce correct samples anyway — it decodes forward and discards
everything before the target — but at roughly 80 M samples/s that means the last
job of a 6-way parallel decode would spend **66 minutes** walking to its own
starting point. In practice `.ldf` had no random access at all; every seek was a
linear scan from the beginning, which is invisible on a small file and fatal on a
large one.

`lddecode/flacseek.py` fixes this. FLAC frames carry their own position: for a
fixed-blocksize stream — which is what `ld-compress` writes — the first sample of
frame N is `N * blocksize`. So the file is seekable without an index:

1. Bisect the file by byte offset, resynchronising on frame headers. Candidates
   are validated by sync word, CRC-8, agreement with STREAMINFO, and by requiring
   the *next* frame to continue the sequence — a lone valid-looking header can
   occur by chance inside compressed audio.
2. Hand the decoder a spliced stream: the original metadata headers followed by
   the file's bytes from that offset. ffmpeg sees a well-formed FLAC file that
   happens to begin part-way through the recording.

Bisection is false-position with a forced plain bisection every other step.
Interpolation alone stalls: when the estimated rate is very close to the truth
the guess lands one byte below the bound, finds the same frame again, and the
bracket shrinks by a byte per iteration. That took 45,568 scans on a 107 MB file.
With the safeguard it takes 5.

Measured on the 240 GB capture:

| requested sample | landed on | error | time |
|---|---|---|---|
| 39,543,780,000 | 39,543,779,328 | 672 samples | 0.00 s |
| 197,718,900,000 | 197,718,898,688 | 1,312 | 0.00 s |
| 355,894,020,000 | 355,894,018,048 | 1,952 | 0.02 s |

Always within one block, effectively instant. Files that cannot be positioned
this way — ogg-wrapped or variable blocksize — fall back to the container seek.

Positions are relative to the file's first frame, so a piece cut out of a longer
capture works too: its frames keep the numbering they had in the original.

## The .ldf length trap

`decode_parallel.py` needs to know how long a capture is in order to split it.
The obvious source lies.

On the 2h45m capture, the container reported **272,629,760 samples (6.8 s)**
against a true **395,437,800,000 (164.8 min)** — out by a factor of ~1450. The
value comes straight from STREAMINFO, where the encoder wrote a wrong count; a
capture past 2³⁶ samples cannot be described there at all.

Trusting it decoded 0.07 % of the tape and reported success. That is the worst
kind of failure: silent, and only detectable by noticing the output is far too
short.

Two defences:

* **Cross-check against file size.** Compressed audio is never larger than the
  samples it encodes, so a claim of 545 MB of samples for a 240 GB file is not a
  claim worth using.
* **Read the capture tool's sidecar.** The DomesDay Duplicator and MISRC write
  `<capture>.json` next to the capture with `captureInfo.durationInMilliseconds`,
  which is authoritative.

Failing both, the tools refuse and ask for `--total_samples` or `--duration`
rather than guessing.

## Parallel decoding on one machine

`decode_parallel.py` cuts the capture into one contiguous span per job, decodes
the spans as independent processes, and stitches the results back together.

On this 6-core machine it is worth about 10 %:

| | time | FPS | vs one process |
|---|---|---|---|
| one process, `-t 1` | 601 s | 2.99 | — |
| one process, `-t 6` | 636 s | 2.83 | 0.95x |
| `--jobs 3` | 554 s | 3.28 | **1.10x** |
| `--jobs 6` | 593 s | 3.06 | 1.02x |

With six jobs running, the CPU is pegged at 100 % while the disk sits near idle,
and each decoder uses about 1.6 cores despite `--threads 1` (numba `parallel=True`
and threaded FFTs). Six jobs ask for ~9.6 cores on six physical ones. The limit
looks like memory bandwidth and shared L3 rather than anything fixable in code —
six copies of a large FFT working set do not fit in 12 MB.

**On a machine with more cores and more memory bandwidth this should scale much
better.** Do not take 1.10x as a property of the approach; it is a property of
this CPU. Splitting across machines (below) sidesteps the problem entirely.

### The seams

Each job re-locks sync at its own starting point, so joins need care:

* Jobs deliberately overrun their span; the merge trims the overlap using
  `fileLoc`, cutting each job where the next one actually started rather than at
  the nominal split point. A decoder asked to seek to sample N may lock on a
  field beginning slightly before N, and cutting at N would drop it from both
  sides.
* Field parity is repaired across every join. Without it every frame after the
  join pairs the wrong two fields.
* A job that produced no fields is skipped over, so its predecessor's overrun
  cannot duplicate the content of the job after it.
* Parts are copied field by field and freed as they are absorbed. Merging used to
  hold a second full copy of the decode on disk — 1,135 GB at the peak for this
  tape, which would have failed at the very end after 22 hours of decoding.
  Peak is now the output plus one part.

Verified on real decoded output: `fileLoc` strictly increasing, no duplicate
positions, parity alternating, luma and chroma byte-exact against the field
count, and the largest gap at a seam no worse than the largest gap the decoder
already produces inside a continuous decode.

### `--no_merge`

Leaves the pieces as separate, individually usable decodes instead of stitching
them, for when there is no room for a merged copy of a whole tape. The overrun is
trimmed in place by truncating the files — no data is rewritten and no extra
space is needed — so the pieces still tile the capture exactly. Concatenating
them is byte-identical to merging.

## Splitting a capture across machines

`split_capture.py` cuts the capture itself into standalone files.

Because FLAC frames are self-contained, a file made of the original headers plus
a run of whole frames is a valid `.ldf` that decodes to that stretch of tape.
Splitting is therefore a **byte copy** — nothing is re-encoded and the pieces are
bit-identical to the corresponding samples of the original. Packed and raw
captures (`.lds`, `.s16`, `.r8`, …) split the same way at sample-group
boundaries.

Each piece carries its own corrected STREAMINFO sample count, and the stream MD5
is zeroed since the original's can never match a piece.

Pieces overlap slightly (2 s by default) because a decoder needs a moment to lock
sync at a cold start. The overlap is recorded in the manifest and trimmed by
`merge_tbc.py`.

Round trip verified on real RF: split a 45 s capture into three, decode each
piece on its own, merge, and compare against decoding the whole thing in one
pass:

| | |
|---|---|
| fields, merged vs single pass | **2694 vs 2696** |
| span covered | 0.01–44.97 s, both |
| worst gap at a seam | identical to the single pass's worst internal gap |

The two-field difference is the parity repair at the two joins.

Unlike parallel decoding on one machine, this scales nearly linearly — each
machine brings its own cores and its own memory bandwidth.

## Tools added

```bash
# Decode with several processes on one machine
python decode_parallel.py vhs --tape_format vhs --system ntsc --jobs 3 \
    capture.ldf output

# ... or leave the pieces separate, to export and delete one at a time
python decode_parallel.py vhs --tape_format vhs --system ntsc --jobs 3 \
    --no_merge capture.ldf output

# Cut a capture into standalone pieces for other machines
python split_capture.py --parts 4 capture.ldf pieces/

# Join the .tbc results, in tape order
python merge_tbc.py --manifest pieces/capture.parts.json --output tape \
    pc1.tbc pc2.tbc pc3.tbc pc4.tbc
```

`decode_parallel.py` shows a live progress bar with frames done, current FPS and
estimated time remaining. It reads progress from the `.tbc` file sizes rather
than the decoders' output, which is block-buffered and lags by a hundred frames.

Keep the `.parts.json` manifest — without it there is no way to know where each
decode belongs or how much overlap to trim.

## When a job stops early

A decoder that hits an unhandled error prints it, saves what it has and **exits
0**.  From the outside the job looks like it finished normally, so the driver
reported "job 1 finished" and carried on.

On a real 2h45m tape that happened 21 minutes into a 55-minute span, and **34
minutes of tape went missing** from the middle of the output.  The only signal
was a gap warning during the merge, hours later.

The crash was a divide by zero in upstream's NTSC burst sync,
`vhsdecode/field.py:_sync_to_burst`:

```python
scale = burst_center_distance / (outlinelen * (burst_center_distance / line_length))
```

`line_length` is zero when two consecutive line locations coincide - a degenerate
field, which a noisy tape produces sooner or later.  The exception propagates out
of `Field.process()` and ends the decode.

Both divisions cancel: the expression is `line_length / outlinelen`.  It is left
as written so the numbers do not shift where it already worked, with a guard that
skips a line carrying no usable timing.  One bad field now costs that field
rather than the rest of the capture.

Separately, `decode_parallel.py` now compares what each job covered against what
it was asked to decode and says so as soon as the jobs finish, instead of leaving
it to surface at merge time or not at all.

Worth recording: a single-process decode would have stopped at the same field and
lost everything after it - 88 minutes instead of 34.  Splitting the work
contained the damage, which is a benefit of parallel decoding that has nothing to
do with speed.

## Filling a gap afterwards

`insert_tbc.py` drops a re-decode of the missing stretch into the hole:

```bash
python insert_tbc.py --into tape.tbc --insert gap.tbc
```

The piece belongs *inside* the file, which `merge_tbc.py` cannot do - it joins
pieces end to end.  Splitting the file and re-merging would need room for a
second copy of the whole decode, hundreds of gigabytes for a full tape.

Instead the file is extended by the size of the insert and the tail is shifted
along, backwards from its end so nothing is overwritten before it has been
copied.  The only extra space needed is the insert itself.  Overrun on both sides
is trimmed by `fileLoc` and field parity is repaired at both new joins.

Metadata is written last: until then the file still matches its old index, so an
interrupted run is recoverable.  `--dry-run` reports what would happen.

## What did not work

**CUDA.** Batched FFTs on an RTX 3090 run about 15x faster than on the CPU, but
whether that helps depends on where the time actually goes, and that differs by
capture. On a synthetic capture the demodulator was not the bottleneck and the
projected gain was 10–25 %. On the real capture `demodblock` is 70–80 % of the
time and mostly FFTs, which makes a GPU path look much more attractive. Not
attempted here; profile your own material before assuming either result.

**More threads.** See the top of this document.
