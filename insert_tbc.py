#!/usr/bin/env python3
"""Drop a decode into a gap in the middle of another decode.

When one job of a parallel decode stops early, its stretch of tape is missing
from the middle of the merged output.  Decoding that stretch again gives a piece
that belongs *inside* the existing file, which merge_tbc.py cannot do - it joins
pieces end to end.

Splitting the file and re-merging would need room for a second copy of the whole
decode, which for a full tape is hundreds of gigabytes.  Instead this extends the
file by the size of the insert and shifts the tail along, so the only extra space
needed is the insert itself.

    python insert_tbc.py --into tape.tbc --insert gap.tbc

Both need their .tbc.json beside them, and their _chroma.tbc if they have one.
Field positions (fileLoc) place the insert, so both must be decodes of the same
capture.

The file is rewritten in place.  It is checked first and the metadata is written
last, so an interrupted run leaves a file whose .json still describes the old
layout - recoverable, but back it up if you can afford to.
"""

import argparse
import json
import os
import shutil
import sys

from lddecode import tbcmerge


def chroma_of(tbc_path):
    root = tbc_path[:-4] if tbc_path.lower().endswith(".tbc") else tbc_path
    candidate = root + "_chroma.tbc"
    return candidate if os.path.exists(candidate) else None


def find_gap(locs, samples_per_field):
    """Index of the field after the largest gap, and the gap's edges."""
    if len(locs) < 2:
        return None
    steps = [(locs[i + 1] - locs[i], i) for i in range(len(locs) - 1)]
    biggest, at = max(steps)
    if biggest < 4 * samples_per_field:
        return None
    return at + 1, locs[at], locs[at + 1]


def plan(into_meta, insert_meta, samples_per_field, say):
    """Which insert fields go in, and where.  Returns (index, [field indices])."""
    into_locs = [f["fileLoc"] for f in into_meta["fields"]]
    ins_locs = [f["fileLoc"] for f in insert_meta["fields"]]

    gap = find_gap(into_locs, samples_per_field)
    if gap is None:
        raise RuntimeError(
            "no gap found in the target - there is nothing to insert into")
    at, gap_start, gap_end = gap

    if not (ins_locs[0] < gap_end and ins_locs[-1] > gap_start):
        raise RuntimeError(
            "the insert covers %.1f s - %.1f s of tape but the gap is %.1f s - "
            "%.1f s; these do not overlap"
            % (ins_locs[0] / 40e6, ins_locs[-1] / 40e6,
               gap_start / 40e6, gap_end / 40e6))

    # Keep only what actually falls in the hole; the decode was asked to start
    # early and run long so it could lock sync, and that overrun is already
    # present on both sides.
    keep = [i for i, loc in enumerate(ins_locs) if gap_start < loc < gap_end]
    if not keep:
        raise RuntimeError("the insert has no fields inside the gap")

    # Field parity has to keep alternating across both new joins, or every frame
    # after them pairs the wrong two fields.
    before = bool(into_meta["fields"][at - 1]["isFirstField"])
    if bool(insert_meta["fields"][keep[0]]["isFirstField"]) == before:
        keep = keep[1:]
        say("  dropping one field at the start of the insert to keep parity")
    after = bool(into_meta["fields"][at]["isFirstField"])
    while keep and bool(insert_meta["fields"][keep[-1]]["isFirstField"]) == after:
        keep = keep[:-1]
        say("  dropping one field at the end of the insert to keep parity")
    if not keep:
        raise RuntimeError("nothing left of the insert after fixing parity")

    return at, keep


def shift_and_write(path, ins_path, field_bytes, at, keep, n_existing, say):
    """Open a hole at field `at` and write the insert's fields into it."""
    count = len(keep)
    hole = count * field_bytes
    tail = n_existing - at            # fields after the insertion point

    with open(path, "r+b") as dst, open(ins_path, "rb") as src:
        # Grow first, then move the tail backwards from its end, so no field is
        # overwritten before it has been copied.
        dst.truncate((n_existing + count) * field_bytes)

        chunk_fields = max(1, (8 * 1024 * 1024) // field_bytes)
        moved = 0
        while moved < tail:
            n = min(chunk_fields, tail - moved)
            src_first = at + tail - moved - n
            dst.seek(src_first * field_bytes)
            data = dst.read(n * field_bytes)
            if len(data) != n * field_bytes:
                raise RuntimeError("short read while shifting the tail")
            dst.seek((src_first + count) * field_bytes)
            dst.write(data)
            moved += n
            if tail:
                say("\r  shifting tail: %5.1f%%" % (100.0 * moved / tail), end="")
        if tail:
            say("\r  shifting tail: done      ")

        # Fill the hole.
        dst.seek(at * field_bytes)
        for n, idx in enumerate(keep):
            src.seek(idx * field_bytes)
            data = src.read(field_bytes)
            if len(data) != field_bytes:
                raise RuntimeError("short read from the insert")
            dst.write(data)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Insert a decode into a gap in another decode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n", 2)[2])
    parser.add_argument("--into", required=True, help="the .tbc with the gap")
    parser.add_argument("--insert", required=True, help="the .tbc to drop into it")
    parser.add_argument("--fps", type=float, default=30000.0 / 1001.0)
    parser.add_argument("--input_freq", type=float, default=40.0,
                        help="capture sample rate in MHz (default 40)")
    parser.add_argument("--dry-run", action="store_true",
                        help="say what would happen and stop")
    args = parser.parse_args(argv)

    sample_freq_hz = args.input_freq * 1e6
    samples_per_field = sample_freq_hz / (args.fps * 2)

    for p in (args.into, args.insert):
        if not os.path.exists(p):
            print("error: no such file: %s" % p, file=sys.stderr)
            return 1

    into_meta = tbcmerge.load_meta(args.into)
    insert_meta = tbcmerge.load_meta(args.insert)
    field_bytes = tbcmerge.field_bytes_of(into_meta)
    if tbcmerge.field_bytes_of(insert_meta) != field_bytes:
        print("error: the two decodes have different field sizes; they are not "
              "from the same capture and system", file=sys.stderr)
        return 1

    def say(msg, end="\n"):
        sys.stdout.write(msg + end)
        sys.stdout.flush()

    try:
        at, keep = plan(into_meta, insert_meta, samples_per_field, say)
    except RuntimeError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1

    into_locs = [f["fileLoc"] for f in into_meta["fields"]]
    n_existing = len(into_meta["fields"])
    on_disk = os.path.getsize(args.into) // field_bytes
    if on_disk < n_existing:
        print("error: %s holds %d fields but its metadata lists %d; refusing to "
              "shift a file that does not match its own index"
              % (args.into, on_disk, n_existing), file=sys.stderr)
        return 1

    into_chroma = chroma_of(args.into)
    ins_chroma = chroma_of(args.insert)
    if bool(into_chroma) != bool(ins_chroma):
        print("error: one decode has chroma and the other does not", file=sys.stderr)
        return 1

    needed = len(keep) * field_bytes * (2 if into_chroma else 1)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(args.into))).free
    say("Inserting %d fields (%.0f frames, %.1f s of tape) at position %d of %d"
        % (len(keep), len(keep) / 2.0,
           len(keep) * samples_per_field / sample_freq_hz, at, n_existing))
    say("  gap in the target : %.1f s .. %.1f s of tape"
        % (into_locs[at - 1] / sample_freq_hz, into_locs[at] / sample_freq_hz))
    say("  space needed      : %.1f GB   free: %.1f GB" % (needed / 1e9, free / 1e9))
    if needed > free:
        print("error: not enough free space", file=sys.stderr)
        return 1
    if args.dry_run:
        say("dry run, nothing written")
        return 0

    say("luma:")
    shift_and_write(args.into, args.insert, field_bytes, at, keep, n_existing, say)
    if into_chroma:
        say("chroma:")
        shift_and_write(into_chroma, ins_chroma, field_bytes, at, keep,
                        n_existing, say)

    # Metadata last: until this is written the file still matches the old index,
    # which is what makes an interrupted run recoverable.
    fields = (into_meta["fields"][:at]
              + [dict(insert_meta["fields"][i]) for i in keep]
              + into_meta["fields"][at:])
    for n, f in enumerate(fields):
        f["seqNo"] = n + 1
    into_meta["fields"] = fields
    into_meta["videoParameters"]["numberOfSequentialFields"] = len(fields)
    with open(args.into + ".json", "w") as fh:
        json.dump(into_meta, fh)

    locs = [f["fileLoc"] for f in fields]
    worst = max(b - a for a, b in zip(locs, locs[1:])) if len(locs) > 1 else 0
    say("\nDone: %d fields (%.0f frames), %.1f s .. %.1f s of tape"
        % (len(fields), len(fields) / 2.0,
           locs[0] / sample_freq_hz, locs[-1] / sample_freq_hz))
    say("  largest remaining gap: %.1f s" % (worst / sample_freq_hz))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
