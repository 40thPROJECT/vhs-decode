#!/usr/bin/env python3
"""Join .tbc decodes of one tape back into a single decode.

Companion to split_capture.py: after each piece of a capture has been decoded,
possibly on a different machine, this puts the results back on one timeline.

    python merge_tbc.py --manifest pieces/capture.parts.json --output tape \\
        from_pc1.tbc from_pc2.tbc from_pc3.tbc

The .tbc files are given in tape order and matched to the manifest's pieces in
that order.  Each decode numbers its fields from the start of its own piece, so
the manifest is what says where that piece sits on the tape; without it the
pieces cannot be placed and the overlap between them cannot be trimmed.

Every .tbc must have its .tbc.json beside it, and its _chroma.tbc too if the
decode produced one.

With no manifest the files are simply joined in the order given, assuming they
already follow on from each other with no overlap - which is what
decode_parallel.py --no_merge leaves behind.
"""

import argparse
import json
import os
import sys

from lddecode import tbcmerge


def chroma_for(tbc_path):
    """The chroma file beside a .tbc, if the decode produced one."""
    root = tbc_path[:-4] if tbc_path.lower().endswith(".tbc") else tbc_path
    candidate = root + "_chroma.tbc"
    return candidate if os.path.exists(candidate) else None


def build_specs(tbc_paths, manifest):
    specs = []
    for i, path in enumerate(tbc_paths):
        if not os.path.exists(path):
            raise RuntimeError("no such file: " + path)
        meta = tbcmerge.load_meta(path)
        origin = 0
        label = os.path.basename(path)
        if manifest:
            piece = manifest["parts"][i]
            origin = piece["startSample"]
            label = "%s (from %s)" % (os.path.basename(path), piece["file"])
        specs.append({
            "tbc": path,
            "chroma": chroma_for(path),
            "meta": meta,
            "origin": origin,
            "label": label,
        })
    return specs


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Join .tbc decodes of one tape into a single decode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n", 2)[2])
    parser.add_argument("tbc", nargs="+", help=".tbc files, in tape order")
    parser.add_argument("--output", "-o", required=True,
                        help="output base name (writes <name>.tbc and .tbc.json)")
    parser.add_argument("--manifest", "-m", default=None,
                        help="the .parts.json written by split_capture.py")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    manifest = None
    if args.manifest:
        try:
            with open(args.manifest) as fh:
                manifest = json.load(fh)
        except (OSError, ValueError) as e:
            print("error: cannot read manifest: %s" % e, file=sys.stderr)
            return 1
        if len(manifest.get("parts", [])) != len(args.tbc):
            print("error: the manifest describes %d piece(s) but %d .tbc file(s) "
                  "were given.  They are matched up in order, so there must be one "
                  "for each - decode the missing piece, or drop the manifest to "
                  "join what you have as-is."
                  % (len(manifest.get("parts", [])), len(args.tbc)), file=sys.stderr)
            return 1

    existing = [args.output + s for s in (".tbc", "_chroma.tbc", ".tbc.json")]
    existing = [p for p in existing if os.path.exists(p)]
    if existing and not args.overwrite:
        print("Existing output found, remove it or pass --overwrite:", file=sys.stderr)
        for p in existing:
            print("\t " + p, file=sys.stderr)
        return 1

    try:
        specs = build_specs(args.tbc, manifest)
    except RuntimeError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1

    sample_freq_hz = (manifest or {}).get("sampleRateHz") or 40e6

    print("Joining %d decode(s):\n" % len(specs))
    for spec in specs:
        locs = tbcmerge.locs_of(spec)
        n = len(locs)
        if n:
            print("  %-44s %6d fields  %8.1f s .. %8.1f s of tape"
                  % (spec["label"], n, locs[0] / sample_freq_hz,
                     locs[-1] / sample_freq_hz))
        else:
            print("  %-44s (no fields)" % spec["label"])

    # Placing pieces out of order would silently interleave the tape.
    starts = [tbcmerge.locs_of(s)[0] for s in specs if tbcmerge.locs_of(s)]
    if starts != sorted(starts):
        print("\nerror: these decodes are not in tape order.  Pass them earliest "
              "first - the order on the command line is what places them.",
              file=sys.stderr)
        return 1

    # A .tbc does not record which capture it came from, so the only way to catch
    # files handed over in the wrong order - which would place each decode on the
    # wrong stretch of tape - is to check that each one is about as long as the
    # piece it has been paired with.
    if manifest:
        for spec, piece in zip(specs, manifest["parts"]):
            expected = piece.get("samples")
            locs = tbcmerge.locs_of(spec)
            if not expected or len(locs) < 2:
                continue
            actual = locs[-1] - locs[0]
            if abs(actual - expected) > max(expected * 0.25, 2 * sample_freq_hz):
                print("\nerror: %s covers %.1f s of tape but %s is %.1f s.  These "
                      "are matched up in the order given, so this usually means "
                      "the .tbc files are in the wrong order or one belongs to a "
                      "different piece."
                      % (os.path.basename(spec["tbc"]), actual / sample_freq_hz,
                         piece["file"], expected / sample_freq_hz), file=sys.stderr)
                return 1

    print()
    nfields, dropped = tbcmerge.merge_parts(specs, args.output)

    print("\nWrote %d fields (%.0f frames) to %s.tbc"
          % (nfields, nfields / 2.0, args.output))
    if os.path.exists(args.output + "_chroma.tbc"):
        print("       %s_chroma.tbc" % args.output)
    print("       %s.tbc.json" % args.output)
    if dropped:
        print("Dropped %d field(s) at the joins to keep field order alternating."
              % dropped)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
