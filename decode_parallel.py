#!/usr/bin/env python3
"""Decode one RF capture with several vhs-decode processes at once.

vhs-decode spends most of its time in a stage that cannot be threaded: each field
has to be sync-detected, line-located and time-base corrected in order, and that
work sits on a single thread no matter what --threads is set to.  On a many-core
machine that leaves most of the CPU idle.

This driver cuts the capture into one contiguous span per job, decodes the spans
as independent processes, and stitches the resulting .tbc/.json back into a single
pair of files.  Scaling is close to linear in the number of jobs until the disk or
the core count runs out.

The seams are the trade-off: each job re-locks sync and re-detects levels at its
own starting point, so a frame at each join may differ slightly from what a single
continuous decode would have produced.  Field parity is repaired across joins, so
interlacing and chroma phase stay correct for the rest of the capture.

Usage mirrors decode.py, with --jobs added:

    python decode_parallel.py vhs --tape_format vhs --system ntsc --jobs 6 \\
        capture.lds output

Options this driver adds:

    --jobs N          decode with N processes at once
    --no_merge        leave the parts as separate, individually usable decodes
                      instead of stitching them.  Merging a whole tape needs
                      room for a second copy of it on disk; with --no_merge each
                      part can be exported and deleted before the next one, and
                      the parts are trimmed so they still tile the capture
                      without overlapping.
    --keep_parts      merge, but keep the parts as well
    --total_samples N, --duration SECONDS
                      state the capture length when it cannot be read from the
                      file itself

Any flag this script does not recognise is passed straight through to decode.py.
The input file and the output base name must be the last two arguments.
"""

import argparse
import json
import os
import subprocess
import sys
import time

from lddecode import tbcmerge

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DECODE_PY = os.path.join(SCRIPT_DIR, "decode.py")

# Frame rates by system, as vhs-decode counts them.
FPS = {
    "ntsc": 30000.0 / 1001.0,
    "palm": 30000.0 / 1001.0,
    "pal": 25.0,
    "paln": 25.0,
    "secam": 25.0,
    "mesecam": 25.0,
}

LOG_TAIL_LINES = 25


class InputSizeError(Exception):
    pass


def sidecar_samples(path, sample_freq_hz):
    """Sample count from a capture tool's .json sidecar, if there is one.

    The DomesDay Duplicator and MISRC write <capture>.json next to the capture
    with the true recording length in it.  For a long .ldf this is the only
    trustworthy source: FLAC's header cannot hold a sample count that large and
    ffmpeg falls back to guessing the duration from the bitrate.
    """
    candidate = os.path.splitext(path)[0] + ".json"
    if not os.path.exists(candidate):
        return None, None
    try:
        # These sidecars start with a small metadata block and can then carry
        # megabytes of per-second telemetry, so read the whole thing only once.
        with open(candidate) as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return None, None

    info = meta.get("captureInfo")
    if not isinstance(info, dict):
        return None, None
    ms = info.get("durationInMilliseconds")
    if not ms:
        return None, None
    return int(round(ms / 1000.0 * sample_freq_hz)), candidate


def count_input_samples(path, sample_freq_hz):
    """Total number of RF samples in the capture."""
    ext = os.path.splitext(path)[1].lower()
    size = os.path.getsize(path)

    if ext == ".lds":
        # 4 ten-bit samples packed into every 5 bytes
        return (size // 5) * 4
    if ext == ".r30":
        # 3 ten-bit samples per 4 bytes
        return (size // 4) * 3
    if ext in (".s16", ".raw", ".r16", ".u16", ".tbc"):
        return size // 2
    if ext in (".r8", ".u8", ".s8"):
        return size
    if ext == ".rf":
        return size // 4
    if ext in (".ldf", ".flac", ".wav", ".oga", ".vhs"):
        try:
            import av
        except ImportError:
            raise InputSizeError(
                "reading the length of %s needs PyAV (pip install av)" % ext
            )
        with av.open(path) as container:
            stream = container.streams.audio[0]
            # Count stored samples rather than going via seconds: .ldf files carry
            # a placeholder sample rate (FLAC cannot express 40 MHz), so any
            # duration in seconds derived from the header is meaningless.
            claimed = None
            if stream.duration is not None and stream.time_base is not None:
                claimed = int(round(float(stream.duration * stream.time_base)
                                    * stream.sample_rate))
            elif container.duration is not None:
                claimed = int(container.duration / 1e6 * stream.sample_rate)

        # Cross-check against the file size before trusting it.  FLAC's header
        # holds at most 2**36 samples, and a capture longer than that makes
        # ffmpeg guess the duration from the bitrate - a guess that came out
        # ~1450x short on a real 2h45m capture, which would have silently
        # decoded the first few seconds of the tape and called it done.
        # Compressed audio is never larger than the samples it encodes, so a
        # claim that fails that test is not a claim worth using.
        if claimed is not None and size <= claimed * 2:
            return claimed

        from_sidecar, sidecar_path = sidecar_samples(path, sample_freq_hz)
        if from_sidecar:
            print("note: the container's length metadata is unusable (it claims "
                  "%s samples for a %.1f GB file); using %s instead"
                  % ("{:,}".format(claimed) if claimed else "nothing",
                     size / 1e9, os.path.basename(sidecar_path)))
            return from_sidecar

        raise InputSizeError(
            "cannot tell how long %s is.  The container claims %s samples, which "
            "is impossible for a %.1f GB file, and there is no usable .json "
            "sidecar next to it.  Pass the true length with --total_samples N "
            "or --duration SECONDS."
            % (path, "{:,}".format(claimed) if claimed else "nothing", size / 1e9)
        )

    raise InputSizeError("unsupported input format for splitting: " + ext)


def check_seekable(path, target_sample):
    """Can the decoder actually jump to `target_sample`, or only walk there?

    Parallel decoding needs random access.  Raw and packed captures seek by byte
    offset and are always fine.  Compressed captures go through the container,
    and a FLAC written without a seek table leaves ffmpeg estimating positions
    from the bitrate.  When that estimate is off, `LoadLDF` still returns the
    right samples - it decodes forward and throws away everything before the
    target - so the decode is correct but each job crawls from the start of the
    file.  On a long capture that is hours per job, so it is worth refusing.

    Returns None if seeking is usable, or a description of the problem.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".ldf", ".flac", ".wav", ".oga", ".vhs"):
        return None
    try:
        import av
    except ImportError:
        return None

    try:
        from lddecode.flacseek import FlacFrameIndex, NotSeekableFlac

        FlacFrameIndex(path)
        # The loader positions by FLAC frame header, which is exact regardless of
        # what the container claims.
        return None
    except (ImportError, NotSeekableFlac):
        pass

    # LoadLDF converts an RF sample offset to container units by dividing by
    # 1000 (the container stores 40 kHz standing in for 40 MHz).
    want_ts = target_sample // 1000
    try:
        with av.open(path) as container:
            stream = container.streams.audio[0]
            container.seek(max(0, want_ts - stream.sample_rate), any_frame=True)
            landed = None
            for packet in container.demux(stream):
                if packet.pts is not None:
                    landed = packet.pts * 1000
                    break
    except Exception as e:
        return "could not probe seeking in %s: %s" % (os.path.basename(path), e)

    if landed is None:
        return "no timestamps in %s, so job start points cannot be found" % (
            os.path.basename(path))

    # A tolerance of a second of RF is far more than a well-formed file needs and
    # far less than a broken one misses by.
    if abs(landed - target_sample) > 40_000_000:
        return (
            "seeking in this file does not work: asking for sample %s lands on "
            "sample %s, %.0fx off.  The decoder would still produce correct "
            "output, but only by decoding the whole file up to each job's start "
            "point, so the last job would read most of the capture before it "
            "began.  This is a raw FLAC with no seek table - ffmpeg has to guess "
            "positions from the bitrate."
            % ("{:,}".format(target_sample), "{:,}".format(landed),
               target_sample / landed if landed else float("inf"))
        )
    return None


def parse_input_freq(value):
    """MHz from a --frequency argument, which may carry a suffix like 8fsc."""
    try:
        return float(value)
    except ValueError:
        # Only pay for importing the decoder's parser when a suffix is actually
        # used; lddecode.utils pulls in numba and takes seconds to import.
        from lddecode.utils import parse_frequency

        return parse_frequency(value)


def parse_known(argv):
    """Pull out the arguments this driver needs; leave the rest for decode.py."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--jobs", "-j", type=int, default=0)
    parser.add_argument("--system", default=None)
    parser.add_argument("--pal", action="store_true")
    parser.add_argument("--ntsc", action="store_true")
    parser.add_argument("--palm", action="store_true")
    parser.add_argument("--threads", "-t", type=int, default=1)
    parser.add_argument("--input_freq", "--frequency", "-f", dest="input_freq",
                        default="40.0")
    parser.add_argument("--keep_parts", action="store_true")
    # Leave the parts as separate, individually usable .tbc sets rather than
    # stitching them: merging a whole tape needs room for a second copy of it.
    parser.add_argument("--no_merge", "--no-merge", dest="no_merge",
                        action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_preflight", action="store_true")
    # Escape hatches for captures whose length cannot be read from the file.
    parser.add_argument("--total_samples", type=int, default=0)
    parser.add_argument("--duration", type=float, default=0.0)
    known, rest = parser.parse_known_args(argv)
    return known, rest


def resolve_system(known):
    if known.system:
        return known.system.lower()
    if known.pal:
        return "pal"
    if known.palm:
        return "palm"
    return "ntsc"


def part_paths(base, index):
    stem = "%s.part%02d" % (base, index)
    return {
        "base": stem,
        "tbc": stem + ".tbc",
        "chroma": stem + "_chroma.tbc",
        "json": stem + ".tbc.json",
        "log": stem + ".log",
    }


def _hms(seconds):
    """Duration as h:mm:ss, or --:--:-- when it is not known yet."""
    if seconds is None or seconds < 0 or seconds != seconds or seconds > 359999:
        return "--:--:--"
    seconds = int(seconds)
    return "%d:%02d:%02d" % (seconds // 3600, (seconds // 60) % 60, seconds % 60)


class Progress:
    """Live progress across every job, measured from the .tbc files themselves.

    The obvious source would be each decoder's own "File Frame N" output, but
    that goes through a redirected stdout and sits in an 8 kB block buffer, so it
    lags by a hundred frames or more.  The .tbc files are written field by field
    with no such delay, so their size is an exact count: fields = bytes / field.

    The field size comes from any part's .tbc.json, which the decoder starts
    dumping within its first few fields.
    """

    # Rate is measured over a trailing window rather than since the start, so the
    # estimate reflects the current pace instead of being dragged down forever by
    # process startup.
    WINDOW_SECONDS = 120.0

    def __init__(self, base, njobs, total_fields, stream=sys.stdout):
        self.base = base
        self.njobs = njobs
        self.total_fields = max(1, int(total_fields))
        self.stream = stream
        self.field_bytes = None
        self.started = time.time()
        self.history = []          # (timestamp, fields done)
        self.live = stream.isatty()
        self.last_drawn = 0.0
        self.last_line_len = 0

    def _find_field_bytes(self):
        for i in range(self.njobs):
            path = part_paths(self.base, i)["json"]
            try:
                with open(path) as fh:
                    vp = json.load(fh)["videoParameters"]
                size = vp["fieldWidth"] * vp["fieldHeight"] * 2
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if size > 0:
                return size
        return None

    def fields_done(self):
        if self.field_bytes is None:
            self.field_bytes = self._find_field_bytes()
            if self.field_bytes is None:
                return None
        total = 0
        for i in range(self.njobs):
            try:
                total += os.path.getsize(part_paths(self.base, i)["tbc"]) // self.field_bytes
            except OSError:
                pass
        return total

    def _rate(self, now, done):
        """Fields per second over the trailing window."""
        self.history.append((now, done))
        cutoff = now - self.WINDOW_SECONDS
        while len(self.history) > 2 and self.history[0][0] < cutoff:
            self.history.pop(0)
        first_t, first_done = self.history[0]
        span = now - first_t
        if span < 5.0 or done <= first_done:
            return None
        return (done - first_done) / span

    def draw(self, note=None, force=False):
        now = time.time()
        if not force and now - self.last_drawn < 1.0:
            return
        self.last_drawn = now

        done = self.fields_done()
        if done is None:
            line = "  starting %d job(s)...  elapsed %s" % (
                self.njobs, _hms(now - self.started))
        else:
            rate = self._rate(now, done)
            # Jobs deliberately overshoot their span and the merge trims the
            # overlap away, so the raw sum can pass the total.  That is an
            # implementation detail; reporting 104% of the tape would just look
            # broken, so the display is capped at what will actually be kept.
            shown = min(done, self.total_fields)
            frac = shown / float(self.total_fields)
            eta = max(0.0, (self.total_fields - done) / rate) if rate else None
            width = 28
            filled = int(round(frac * width))
            bar = "#" * filled + "-" * (width - filled)
            line = ("  [%s] %5.1f%%  %6.0f/%.0f frames  %.2f FPS  elapsed %s  left %s"
                    % (bar, frac * 100, shown / 2.0, self.total_fields / 2.0,
                       (rate / 2.0) if rate else 0.0,
                       _hms(now - self.started), _hms(eta)))

        if note:
            self._clear()
            print(note, file=self.stream)
        if self.live:
            self.stream.write("\r" + line.ljust(self.last_line_len))
            self.stream.flush()
            self.last_line_len = len(line)
        elif note or force or now - getattr(self, "_last_plain", 0) > 60:
            # Not a terminal (piped to a file): a carriage-return bar would be
            # unreadable, so log a line now and then instead.
            self._last_plain = now
            print(line, file=self.stream, flush=True)

    def _clear(self):
        if self.live and self.last_line_len:
            self.stream.write("\r" + " " * self.last_line_len + "\r")
            self.stream.flush()
            self.last_line_len = 0

    def finish(self):
        self.draw(force=True)
        self._clear()
        if self.live:
            print(file=self.stream)


def tail_of(path, nlines=LOG_TAIL_LINES):
    try:
        with open(path, "r", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError as e:
        return ["(could not read %s: %s)" % (path, e)]
    lines = [l for l in lines if l.strip()]
    return lines[-nlines:] if lines else ["(the job produced no output at all)"]


def preflight(tape_format_argv):
    """Check the child interpreter can actually start a decode.

    Every job is a separate `python decode.py` and they all fail the same way if
    the checkout is incomplete - a missing Cython extension, a missing vhsd_rust,
    an uninstalled dependency.  Finding that out once, up front, beats launching
    six processes that all die on the same import.
    """
    cmd = [sys.executable, DECODE_PY] + tape_format_argv + ["--help"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True, errors="replace")
    except OSError as e:
        return "could not run %s: %s" % (DECODE_PY, e)
    if proc.returncode != 0:
        out = (proc.stdout or "").strip()
        return ("`%s %s %s --help` failed:\n%s"
                % (os.path.basename(sys.executable), os.path.basename(DECODE_PY),
                   " ".join(tape_format_argv), out))
    return None


def run_jobs(decoder_args, tape_format_argv, in_file, base, bounds, frames_per_job,
             threads, cap_last=False, total_frames=0):
    """Launch one decode process per span and wait for all of them."""
    njobs = len(bounds) - 1
    progress = Progress(base, njobs, total_frames * 2)
    procs = []
    for i, start in enumerate(bounds[:-1]):
        paths = part_paths(base, i)
        cmd = [sys.executable, DECODE_PY] + tape_format_argv + decoder_args + [
            "--threads", str(threads),
            "--start_fileloc", str(int(start)),
        ]
        if i < njobs - 1 or cap_last:
            # Ask for more frames than the span needs.  The merge trims the
            # overrun away, so overshooting only wastes a little work, while
            # undershooting - which happens when a noisy stretch makes the
            # decoder skip - would leave a hole in the middle of the tape.
            # The last job normally has nothing after it to overrun into, so it
            # runs to the end of the file: a low frame estimate must not truncate
            # the tape.  When the caller stated the length outright, that is the
            # range they asked for and the last job is held to it.
            cmd += ["-l", str(int(frames_per_job * 1.02 + 8))]
        cmd += ["--overwrite", in_file, paths["base"]]

        log = open(paths["log"] + ".driver", "w")
        procs.append(
            [i, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log, None]
        )
        print("  job %d: samples %d.. -> %s"
              % (i, start, os.path.basename(paths["base"])))

    started = time.time()
    pending = len(procs)
    try:
        while pending:
            time.sleep(1.0)
            for entry in procs:
                if entry[3] is not None:
                    continue
                rc = entry[1].poll()
                if rc is None:
                    continue
                entry[3] = rc
                entry[2].close()
                pending -= 1
                progress.draw(
                    note="  job %d %s after %s (%d still running)"
                    % (entry[0], "finished" if rc == 0 else "FAILED (exit %d)" % rc,
                       _hms(time.time() - started), pending),
                    force=True,
                )
            progress.draw()
    except KeyboardInterrupt:
        # Without this the decoders outlive the driver: they keep burning every
        # core and holding their output files open, so the next run cannot even
        # delete them.
        progress.finish()
        print("\nInterrupted - stopping %d job(s)..." % pending, file=sys.stderr)
        for _i, proc, log, rc in procs:
            if rc is None:
                proc.terminate()
        for _i, proc, log, rc in procs:
            if rc is None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            log.close()
        raise
    progress.finish()

    return [(i, rc) for i, _proc, _log, rc in procs if rc != 0]


def read_parts(base, njobs):
    """Every part's metadata, checked for what the driver needs from it."""
    parts = []
    for i in range(njobs):
        paths = part_paths(base, i)
        if not os.path.exists(paths["json"]):
            raise RuntimeError("job %d produced no output (%s missing)"
                               % (i, paths["json"]))
        parts.append(tbcmerge.load_meta(paths["tbc"]))
    return parts


def part_specs(base, njobs, parts):
    """The parts in the form lddecode.tbcmerge wants them."""
    specs = []
    for i, part in enumerate(parts):
        paths = part_paths(base, i)
        specs.append({
            "tbc": paths["tbc"],
            "chroma": paths["chroma"] if os.path.exists(paths["chroma"]) else None,
            "meta": part,
            "origin": 0,          # all jobs read the same file, so fileLoc is absolute
            "label": os.path.basename(paths["base"]),
        })
    return specs


def merge(base, njobs, bounds, keep_parts, sample_freq_hz, samples_per_field):
    """Stitch the parts into <base>.tbc / <base>_chroma.tbc / <base>.tbc.json."""
    parts = read_parts(base, njobs)
    specs = part_specs(base, njobs, parts)

    def drop_part(i):
        if keep_parts:
            return
        # Free this part's pixel data as soon as it has been copied.  Waiting
        # until every part was merged meant holding a second full copy of the
        # decode on disk at the peak - for a whole tape that is hundreds of
        # gigabytes of headroom needed for nothing.  The .json and the log are
        # kept until the end; they are tiny and useful if this fails.
        paths = part_paths(base, i)
        for path in (paths["tbc"], paths["chroma"]):
            try:
                os.remove(path)
            except OSError:
                pass

    # Warn about tape that no job covered, before the parts are consumed.
    cut_at = tbcmerge.cut_points(specs)
    for i, spec in enumerate(specs):
        locs = tbcmerge.locs_of(spec)
        if not locs or cut_at[i] is None:
            continue
        keep = tbcmerge.kept_indices(locs, cut_at[i])
        if keep and cut_at[i] - locs[keep[-1]] > 4 * samples_per_field:
            print("  warning: gap of ~%.1f s between job %d and the next"
                  % ((cut_at[i] - locs[keep[-1]]) / sample_freq_hz, i))

    nfields, dropped = tbcmerge.merge_parts(specs, base, on_part_done=drop_part)

    if not keep_parts:
        for i in range(njobs):
            paths = part_paths(base, i)
            for key in ("tbc", "chroma", "json", "log"):
                for candidate in (paths[key], paths[key] + ".driver"):
                    if os.path.exists(candidate):
                        os.remove(candidate)

    return nfields, dropped


def trim_parts(base, njobs, sample_freq_hz):
    """Cut each part's overrun so the parts tile the capture without overlapping.

    Used instead of merging.  Each job decodes a little past its span so a noisy
    stretch cannot leave a hole, and normally the merge drops that overrun.  When
    the parts are kept as separate files they need the same trim, or the tape
    repeats a second or two of itself at every join.

    The overrun is always at the end of a part, so this is a truncate: no data is
    rewritten and no extra disk space is needed.  What is left concatenates to
    exactly what a merge would have produced.
    """
    parts = read_parts(base, njobs)
    specs = part_specs(base, njobs, parts)
    cut_at = tbcmerge.cut_points(specs)

    summary = []
    for i in range(njobs):
        paths = part_paths(base, i)
        part = parts[i]
        fields = part["fields"]
        keep = tbcmerge.kept_indices(tbcmerge.locs_of(specs[i]), cut_at[i])
        field_bytes = tbcmerge.field_bytes_of(part)

        if len(keep) < len(fields):
            for path in (paths["tbc"], paths["chroma"]):
                if not os.path.exists(path):
                    continue
                wanted = len(keep) * field_bytes
                if os.path.getsize(path) > wanted:
                    with open(path, "r+b") as fh:
                        fh.truncate(wanted)
            part["fields"] = [fields[idx] for idx in keep]
            part["videoParameters"]["numberOfSequentialFields"] = len(keep)
            for n, f in enumerate(part["fields"]):
                f["seqNo"] = n + 1
            with open(paths["json"], "w") as fh:
                json.dump(part, fh)

        if keep:
            start = fields[keep[0]]["fileLoc"] / sample_freq_hz
            end = fields[keep[-1]]["fileLoc"] / sample_freq_hz
        else:
            start = end = 0.0
        summary.append((i, len(keep), len(fields) - len(keep), start, end))
    return summary


def check_coverage(base, njobs, bounds, sample_freq_hz, tolerance=0.9):
    """Jobs whose output covers much less tape than they were asked to decode.

    Returns [(job, seconds covered, seconds expected), ...] for the short ones.
    """
    short = []
    for i in range(njobs):
        paths = part_paths(base, i)
        try:
            with open(paths["json"]) as fh:
                fields = json.load(fh).get("fields") or []
        except (OSError, ValueError):
            continue
        if not fields:
            short.append((i, 0.0, (bounds[i + 1] - bounds[i]) / sample_freq_hz))
            continue
        locs = [f["fileLoc"] for f in fields if "fileLoc" in f]
        if not locs:
            continue
        covered = (locs[-1] - locs[0]) / sample_freq_hz
        expected = (bounds[i + 1] - bounds[i]) / sample_freq_hz
        if expected > 0 and covered < expected * tolerance:
            short.append((i, covered, expected))
    return short


def split_files(rest):
    """The input file and output base name, which must be the last two arguments."""
    if len(rest) < 2:
        return None, None, "need an input file and an output base name"
    in_file, out_base = rest[-2], rest[-1]
    if in_file.startswith("-") or out_base.startswith("-"):
        return None, None, (
            "the input file and the output base name must be the last two "
            "arguments (got %r %r)" % (in_file, out_base)
        )
    if not os.path.isfile(in_file):
        return None, None, "input file not found: %s" % in_file
    return in_file, out_base, None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 1

    # decode.py takes the tape type as its first positional argument.
    tape_format_argv = []
    if argv and not argv[0].startswith("-"):
        tape_format_argv = [argv.pop(0)]

    known, rest = parse_known(argv)

    for conflicting in ("-s", "--start", "-l", "--length", "--start_fileloc"):
        if conflicting in rest:
            print("error: %s cannot be combined with parallel decoding - this driver "
                  "sets the range of each job itself" % conflicting, file=sys.stderr)
            return 1

    in_file, out_base, err = split_files(rest)
    if err:
        print("error: " + err, file=sys.stderr)
        return 1
    rest = rest[:-2]

    jobs = known.jobs
    if jobs <= 0:
        # Each job saturates roughly one and a half cores once its demod threads
        # are counted, so half the logical cores is a reasonable default.
        jobs = max(1, (os.cpu_count() or 4) // 2)

    system = resolve_system(known)
    fps = FPS.get(system, FPS["ntsc"])
    try:
        sample_freq_hz = parse_input_freq(known.input_freq) * 1e6
    except ValueError:
        print("error: could not read a sample rate from -f %s" % known.input_freq,
              file=sys.stderr)
        return 1

    if known.total_samples:
        total_samples = known.total_samples
    elif known.duration:
        total_samples = int(round(known.duration * sample_freq_hz))
    else:
        try:
            total_samples = count_input_samples(in_file, sample_freq_hz)
        except (InputSizeError, OSError) as e:
            print("error: %s" % e, file=sys.stderr)
            return 1

    total_seconds = total_samples / sample_freq_hz
    total_frames = total_seconds * fps
    if jobs > 1 and total_frames / jobs < 30:
        jobs = max(1, int(total_frames // 30))
        print("capture is short, reducing to %d job(s)" % jobs)

    existing = [out_base + s for s in (".tbc", "_chroma.tbc", ".tbc.json")]
    existing = [p for p in existing if os.path.exists(p)]
    if existing and not known.overwrite:
        print("Existing decode files found, remove them or run with --overwrite",
              file=sys.stderr)
        for p in existing:
            print("\t " + p, file=sys.stderr)
        return 1

    bounds = [int(round(i * total_samples / jobs)) for i in range(jobs)] + [total_samples]
    frames_per_job = total_frames / jobs

    # Rebuild the system flag for the child processes.
    passthrough = list(rest)
    if known.system:
        passthrough += ["--system", known.system]
    elif known.pal:
        passthrough += ["--pal"]
    elif known.palm:
        passthrough += ["--palm"]
    elif known.ntsc:
        passthrough += ["--ntsc"]
    if str(known.input_freq) != "40.0":
        passthrough += ["-f", str(known.input_freq)]

    print("Input:  %s (%.1f s, ~%.0f frames at %.3f MSPS)"
          % (in_file, total_seconds, total_frames, sample_freq_hz / 1e6))

    if jobs > 1:
        problem = check_seekable(in_file, bounds[-2])
        if problem:
            print("\nerror: %s\n" % problem, file=sys.stderr)
            print("Options: decode this capture sequentially with decode.py, or "
                  "convert it to a format that seeks by byte offset (.lds / .s16) "
                  "and split that instead.  --jobs 1 also skips this check.",
                  file=sys.stderr)
            return 1

    if not known.skip_preflight:
        problem = preflight(tape_format_argv)
        if problem:
            print("\nerror: this checkout cannot run a decode, so the jobs would all "
                  "fail the same way:\n\n%s\n" % problem, file=sys.stderr)
            print("A source checkout needs the Cython extensions built "
                  "(python setup.py build_ext --inplace) and, if vhsd_rust is "
                  "missing, either a cargo build or vhsdecode/rust_fallback.py.",
                  file=sys.stderr)
            return 1

    print("Splitting into %d jobs of ~%.0f frames, %d decode thread(s) each"
          % (jobs, frames_per_job, known.threads))

    started = time.time()
    failed = run_jobs(passthrough, tape_format_argv, in_file, out_base, bounds,
                      frames_per_job, known.threads,
                      cap_last=bool(known.total_samples or known.duration),
                      total_frames=total_frames)
    if failed:
        print("", file=sys.stderr)
        for i, rc in failed:
            log = part_paths(out_base, i)["log"] + ".driver"
            print("job %d failed with exit code %d, last lines of its output:"
                  % (i, rc), file=sys.stderr)
            for line in tail_of(log):
                print("    " + line, file=sys.stderr)
            print("  (full log: %s)\n" % log, file=sys.stderr)
        return 1

    decode_time = time.time() - started

    # A decoder that hits an unhandled error part-way through prints it, saves
    # what it has and exits 0, so "the job finished" says nothing about whether
    # it covered its span.  Say so here rather than letting it surface hours
    # later as a gap warning during the merge, or not at all.
    short = check_coverage(out_base, jobs, bounds, sample_freq_hz)
    if short:
        print("", file=sys.stderr)
        for i, covered, expected in short:
            print("WARNING: job %d covered only %s of its %s span - %s of tape is "
                  "missing from the output.  Its decoder stopped early; the end of "
                  "%s.log.driver usually says why."
                  % (i, _hms(covered), _hms(expected), _hms(expected - covered),
                     os.path.basename(part_paths(out_base, i)["base"])),
                  file=sys.stderr)
        print("", file=sys.stderr)

    if known.no_merge:
        summary = trim_parts(out_base, jobs, sample_freq_hz)
        kept = sum(n for _i, n, _t, _s, _e in summary)
        print("\nDecoded %d fields (%.0f frames) in %s -> %.2f FPS"
              % (kept, kept / 2.0, _hms(decode_time),
                 (kept / 2.0) / decode_time if decode_time else 0.0))
        print("Left as %d separate decodes (not merged):\n" % jobs)
        for i, n, trimmed, start, end in summary:
            print("  %-28s %6d fields (%5.0f frames)  %s .. %s of tape%s"
                  % (os.path.basename(part_paths(out_base, i)["base"]), n, n / 2.0,
                     _hms(start), _hms(end),
                     "  [%d overlapping trimmed]" % trimmed if trimmed else ""))
        print("\nThe parts do not overlap and follow on from each other, so each "
              "can be exported on its own and deleted before the next - which is "
              "the point of --no_merge: a merged tape needs room for a second "
              "copy of itself.")
        print("Export one with:  tbc-video-export %s.tbc"
              % part_paths(out_base, 0)["base"])
        return 0

    nfields, dropped = merge(out_base, jobs, bounds, known.keep_parts,
                             sample_freq_hz, sample_freq_hz / (fps * 2))
    frames = nfields / 2.0
    total_time = time.time() - started

    print("\nDecoded %d fields (%.0f frames) in %.1f s decode + %.1f s merge "
          "-> %.2f FPS overall"
          % (nfields, frames, decode_time, total_time - decode_time,
             frames / total_time if total_time else 0.0))
    if dropped:
        print("Dropped %d field(s) at the joins to keep field order alternating" % dropped)
    print("Wrote %s.tbc, %s_chroma.tbc and %s.tbc.json" % (out_base, out_base, out_base))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
