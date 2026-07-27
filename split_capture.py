#!/usr/bin/env python3
"""Cut an RF capture into standalone pieces, to decode on several machines.

A .ldf is a FLAC stream, and FLAC frames are self-contained: a file made of the
original headers followed by a run of whole frames is a perfectly valid .ldf that
decodes to that stretch of the tape.  So splitting is a byte copy - nothing is
re-encoded, nothing is re-compressed, and the pieces are bit-identical to the
corresponding samples of the original.  Packed and raw captures (.lds, .s16 and
friends) split the same way, at sample boundaries.

Each piece overlaps the next by a little.  A decoder needs a moment to lock sync
at a cold start, so without an overlap a field or two would be lost at every
join.  The overlap is recorded in the manifest and `merge_tbc.py` trims it back
out, so the decoded pieces still tile the tape exactly.

    python split_capture.py --parts 4 capture.ldf pieces/

writes pieces/capture.part00.ldf .. part03.ldf plus capture.parts.json, the
manifest that says where each piece sits on the tape.  Copy a piece to each
machine, decode it there however you like, bring the .tbc files back, then:

    python merge_tbc.py --manifest pieces/capture.parts.json --output tape \\
        machine1.tbc machine2.tbc machine3.tbc machine4.tbc
"""

import argparse
import json
import os
import shutil
import sys
import time

from decode_parallel import (
    FPS, InputSizeError, count_input_samples, parse_input_freq, _hms,
)

# Bytes per sample for the formats that can be cut at a sample boundary.
RAW_LAYOUTS = {
    ".s16": (2, 1), ".raw": (2, 1), ".r16": (2, 1), ".u16": (2, 1),
    ".r8": (1, 1), ".u8": (1, 1), ".s8": (1, 1),
    ".rf": (4, 1),
    ".lds": (5, 4),          # 4 ten-bit samples per 5 bytes
    ".r30": (4, 3),          # 3 ten-bit samples per 4 bytes
}

FLAC_EXTS = (".ldf", ".flac", ".oga", ".vhs")


def human(n):
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return "%.1f %s" % (n, unit)
        n /= 1024.0


def copy_span(src, dst, start_byte, end_byte, header=b"", progress=None):
    """Copy src[start_byte:end_byte] to dst, after an optional header."""
    chunk = 8 * 1024 * 1024
    written = 0
    with open(dst, "wb") as out:
        if header:
            out.write(header)
        src.seek(start_byte)
        remaining = end_byte - start_byte
        while remaining > 0:
            data = src.read(min(chunk, remaining))
            if not data:
                break
            out.write(data)
            remaining -= len(data)
            written += len(data)
            if progress:
                progress(len(data))
    return written


def restamp_flac_header(header, total_samples):
    """Give a piece's headers its own sample count, and drop the stream MD5.

    A piece inherits the source's STREAMINFO, which describes the whole capture -
    so every piece would claim the length of the original.  On a capture long
    enough for that field to be wrong in the first place, tools then have no way
    at all to tell how long a piece is.  The MD5 covers the original samples and
    can never match a piece, and all-zero is FLAC's "not computed".

    STREAMINFO sits at file offset 8 (after 'fLaC' and the 4-byte block header).
    Its last 8 bytes before the MD5 pack sample rate, channels, bit depth and a
    36-bit total sample count; only the count changes here.
    """
    body = 8
    if len(header) < body + 34 or header[:4] != b"fLaC":
        return header
    out = bytearray(header)
    packed = int.from_bytes(out[body + 10:body + 18], "big")
    packed = (packed & ~((1 << 36) - 1)) | (min(total_samples, (1 << 36) - 1))
    out[body + 10:body + 18] = packed.to_bytes(8, "big")
    out[body + 18:body + 34] = b"\0" * 16
    return bytes(out)


def plan_flac(path, stem, cuts):
    """Resolve each span to whole FLAC frames.

    A piece's declared start has to be the sample its decoder will actually
    produce first, since that is what places the decode on the tape, so the
    boundaries are snapped to real frame headers before anything is copied.
    """
    from lddecode.flacseek import FlacFrameIndex

    index = FlacFrameIndex(path)
    ext = os.path.splitext(path)[1]
    plan = []
    for i, (start, end) in enumerate(cuts):
        byte_a, first_sample = index.find_frame(start)
        if end is None:
            byte_b, end_sample = index.size, None
        else:
            byte_b, end_sample = index.find_frame(end)
        # Frame numbering is relative to the source's first frame, so `origin`
        # is where this piece really sits on the tape.
        origin = index.base_sample + first_sample
        if end_sample is None:
            # Runs to EOF; estimate from the bytes, only to stamp the header.
            per_byte = (first_sample / float(byte_a - index.first_frame_offset)
                        if byte_a > index.first_frame_offset else 0)
            span = int((byte_b - byte_a) * per_byte) if per_byte else 0
        else:
            span = end_sample - first_sample
        plan.append({
            "file": "%s.part%02d%s" % (stem, i, ext),
            "startSample": origin,
            "samples": span,
            "byte_a": byte_a,
            "byte_b": byte_b,
            "header": restamp_flac_header(index.header_bytes, span),
        })
    return plan


def plan_raw(path, stem, cuts):
    """Resolve each span to whole sample groups, so packed samples stay intact."""
    ext = os.path.splitext(path)[1].lower()
    group_bytes, group_samples = RAW_LAYOUTS[ext]
    total = os.path.getsize(path)

    def to_byte(sample):
        return min(total, (sample // group_samples) * group_bytes)

    plan = []
    for i, (start, end) in enumerate(cuts):
        byte_a = to_byte(start)
        byte_b = total if end is None else to_byte(end)
        plan.append({
            "file": "%s.part%02d%s" % (stem, i, ext),
            "startSample": (byte_a // group_bytes) * group_samples,
            "samples": ((byte_b - byte_a) // group_bytes) * group_samples,
            "byte_a": byte_a,
            "byte_b": byte_b,
            "header": b"",
        })
    return plan


def write_pieces(path, out_dir, plan, progress):
    pieces = []
    with open(path, "rb") as src:
        for entry in plan:
            dst = os.path.join(out_dir, entry["file"])
            size = copy_span(src, dst, entry["byte_a"], entry["byte_b"],
                             entry["header"], progress)
            pieces.append({
                "file": entry["file"],
                "startSample": entry["startSample"],
                "samples": entry["samples"],
                "bytes": size + len(entry["header"]),
            })
    return pieces


def plan_cuts(total_samples, parts, overlap):
    """(start, end) sample spans, each overlapping the one before by `overlap`.

    Returns (cuts, overlap actually used), both in samples.  An overlap
    approaching the size of a piece would have every piece starting near the
    front of the capture, copying the same stretch of tape over and over, so it
    is capped at a quarter of a piece.
    """
    piece = total_samples / float(parts)
    capped = max(0, min(overlap, int(piece // 4)))

    cuts = []
    for i in range(parts):
        nominal = int(round(i * piece))
        start = max(0, nominal - capped) if i else 0
        end = None if i == parts - 1 else int(round((i + 1) * piece))
        cuts.append((start, end))
    return cuts, capped


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Cut an RF capture into standalone pieces for decoding elsewhere.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n", 2)[2])
    parser.add_argument("input", help="the capture to split (.ldf, .lds, .s16, ...)")
    parser.add_argument("output_dir", help="where to write the pieces")
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument("--parts", type=int, default=0,
                      help="cut into this many equal pieces")
    size.add_argument("--minutes", type=float, default=0.0,
                      help="cut into pieces of about this many minutes of tape")
    parser.add_argument("--overlap", type=float, default=2.0,
                        help="seconds of tape each piece repeats from the one "
                             "before, so the decoder can lock sync (default 2)")
    parser.add_argument("--system", default="ntsc",
                        help="video system, only used to report frame counts")
    parser.add_argument("--input_freq", "--frequency", "-f", dest="input_freq",
                        default="40.0", help="sample rate in MHz (default 40)")
    parser.add_argument("--total_samples", type=int, default=0,
                        help="capture length, when it cannot be read from the file")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="capture length in seconds, same purpose")
    args = parser.parse_args(argv)

    ext = os.path.splitext(args.input)[1].lower()
    if ext not in FLAC_EXTS and ext not in RAW_LAYOUTS:
        print("error: don't know how to split %s files" % ext, file=sys.stderr)
        return 1
    if not os.path.isfile(args.input):
        print("error: no such file: %s" % args.input, file=sys.stderr)
        return 1

    try:
        sample_freq_hz = parse_input_freq(args.input_freq) * 1e6
    except ValueError:
        print("error: bad sample rate: %s" % args.input_freq, file=sys.stderr)
        return 1

    if args.total_samples:
        total_samples = args.total_samples
    elif args.duration:
        total_samples = int(round(args.duration * sample_freq_hz))
    else:
        try:
            total_samples = count_input_samples(args.input, sample_freq_hz)
        except (InputSizeError, OSError) as e:
            print("error: %s" % e, file=sys.stderr)
            return 1

    seconds = total_samples / sample_freq_hz
    fps = FPS.get(args.system.lower(), FPS["ntsc"])

    parts = args.parts
    if not parts:
        parts = max(1, int(round(seconds / (args.minutes * 60.0))))
    if parts < 1:
        print("error: that works out to less than one piece", file=sys.stderr)
        return 1

    wanted_overlap = int(round(args.overlap * sample_freq_hz))
    cuts, overlap = plan_cuts(total_samples, parts, wanted_overlap)
    if args.total_samples or args.duration:
        # The last piece normally runs to the end of the file, so a length read
        # from the capture cannot cut the tape short.  When the length was stated
        # outright, that is the range being asked for and the last piece has to
        # respect it - otherwise asking for 45 seconds of a 224 GB capture copies
        # all 224 GB.
        cuts[-1] = (cuts[-1][0], total_samples)
    if overlap < wanted_overlap:
        print("note: an overlap of %.1f s does not fit pieces of %s; using %.1f s"
              % (args.overlap, _hms(seconds / parts), overlap / sample_freq_hz))

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    stem = os.path.splitext(os.path.basename(args.input))[0]

    src_size = os.path.getsize(args.input)
    print("Input:  %s (%s, %s of tape, ~%.0f frames)"
          % (args.input, human(src_size), _hms(seconds), seconds * fps))

    try:
        if ext in FLAC_EXTS:
            plan = plan_flac(args.input, stem, cuts)
        else:
            plan = plan_raw(args.input, stem, cuts)
    except Exception as e:
        print("error: cannot work out where to cut: %s" % e, file=sys.stderr)
        return 1

    # Progress is measured against what will actually be written, not the size of
    # the source: asking for a slice of a 224 GB capture copies a fraction of it,
    # and a bar counting up to the whole file would be meaningless.
    to_write = sum(e["byte_b"] - e["byte_a"] + len(e["header"]) for e in plan)
    print("Cutting into %d piece(s) of ~%s, overlapping %.1f s - writing %s\n"
          % (parts, _hms(seconds / parts), overlap / sample_freq_hz, human(to_write)))

    started = time.time()
    copied = [0]
    last = [0.0]

    def progress(n):
        copied[0] += n
        now = time.time()
        if now - last[0] < 1.0:
            return
        last[0] = now
        frac = copied[0] / float(to_write) if to_write else 0
        rate = copied[0] / max(1e-9, now - started)
        eta = (to_write - copied[0]) / rate if rate else None
        width = 28
        filled = int(round(min(1.0, frac) * width))
        sys.stdout.write("\r  [%s] %5.1f%%  %s of %s  %s/s  left %s   "
                         % ("#" * filled + "-" * (width - filled), frac * 100,
                            human(copied[0]), human(to_write), human(rate), _hms(eta)))
        sys.stdout.flush()

    try:
        pieces = write_pieces(args.input, args.output_dir, plan, progress)
    except KeyboardInterrupt:
        print("\nInterrupted - the pieces written so far are still valid, but the "
              "manifest was not written.", file=sys.stderr)
        return 130
    sys.stdout.write("\r" + " " * 78 + "\r")

    manifest = {
        "source": os.path.basename(args.input),
        "sourceBytes": src_size,
        "sourceSamples": total_samples,
        "sampleRateHz": sample_freq_hz,
        "system": args.system.lower(),
        "overlapSamples": overlap,
        "parts": pieces,
    }
    manifest_path = os.path.join(args.output_dir, stem + ".parts.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("Wrote %d piece(s) in %s:\n" % (len(pieces), _hms(time.time() - started)))
    for i, piece in enumerate(pieces):
        start_s = piece["startSample"] / sample_freq_hz
        end_s = (pieces[i + 1]["startSample"] / sample_freq_hz
                 if i + 1 < len(pieces) else seconds)
        print("  %-34s %9s   %s .. %s of tape"
              % (piece["file"], human(piece["bytes"]), _hms(start_s), _hms(end_s)))
    print("\nManifest: %s" % manifest_path)
    print("Keep it - merge_tbc.py needs it to know where each decode belongs.")
    print("\nDecode a piece on any machine, e.g.:")
    print("  python decode.py vhs --tape_format vhs --system %s %s out"
          % (args.system.lower(), pieces[0]["file"]))
    print("\nThen bring the .tbc files back and run:")
    print("  python merge_tbc.py --manifest %s --output tape <tbc> <tbc> ..."
          % os.path.basename(manifest_path))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
