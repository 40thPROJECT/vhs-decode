"""Stitch several .tbc decodes of one tape back into a single pair of files.

Used by both parallel decoding on one machine (decode_parallel.py) and decoding
split captures on several machines (merge_tbc.py).  The awkward part is the same
either way: neighbouring decodes overlap, each one re-locks sync at its own
starting point, and field parity has to keep alternating across every join or
every frame after it is assembled from the wrong pair of fields.

Each input is described by a dict:

    tbc      path to the luma .tbc
    chroma   path to the chroma .tbc, or None
    meta     the parsed .tbc.json
    origin   absolute sample offset of this decode within the whole capture,
             added to every fileLoc so parts decoded from separately split
             files can be placed on the same timeline
    label    something to call it in messages
"""

import json
import os


def load_meta(tbc_path):
    """The .tbc.json beside a .tbc, checked for what merging needs from it."""
    json_path = tbc_path + ".json"
    if not os.path.exists(json_path):
        raise RuntimeError("no metadata beside %s (expected %s)"
                           % (tbc_path, os.path.basename(json_path)))
    with open(json_path) as fh:
        meta = json.load(fh)
    fields = meta.get("fields") or []
    for f in fields:
        if "fileLoc" not in f:
            raise RuntimeError(
                "%s has fields without a fileLoc; merging needs it to know where "
                "each field came from" % os.path.basename(json_path)
            )
    meta["fields"] = fields
    return meta


def field_bytes_of(meta):
    vp = meta["videoParameters"]
    return vp["fieldWidth"] * vp["fieldHeight"] * 2


def locs_of(spec):
    """This decode's field positions, on the whole capture's timeline."""
    origin = spec.get("origin", 0)
    return [f["fileLoc"] + origin for f in spec["meta"]["fields"]]


def cut_points(specs):
    """Where each part stops, as an absolute sample offset (None for the last).

    Cut each part where the next one that actually produced fields started,
    rather than at a nominal split point: a decoder starting at sample N may lock
    on a field that begins slightly before N, and cutting at N would drop it from
    both sides, leaving a hole.  Skipping over empty parts matters too - cutting
    at an empty part's bound would let the previous part's overrun duplicate the
    content of the part after it.
    """
    cut_at = [None] * len(specs)
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            following = locs_of(specs[j])
            if following:
                cut_at[i] = following[0]
                break
    return cut_at


def kept_indices(locs, limit):
    """The fields of a part that belong to it rather than to the next one."""
    keep = []
    for idx, loc in enumerate(locs):
        if limit is not None and loc >= limit:
            break
        keep.append(idx)
    return keep


def _open_checked(path, field_bytes, nfields, label, say):
    """Open a .tbc and check it holds as many fields as its metadata claims.

    A decode that was killed part-way can leave a JSON listing fields that never
    reached the disk; copying those would emit truncated or empty fields.
    """
    fh = open(path, "rb")
    have = os.path.getsize(path) // field_bytes
    if have < nfields:
        say("  warning: %s holds %d field(s) but its metadata lists %d - using "
            "what is on disk" % (os.path.basename(path), have, nfields))
    return fh, have


def merge_parts(specs, out_base, say=print, on_part_done=None):
    """Write out_base.tbc / _chroma.tbc / .tbc.json from `specs`, in order.

    `on_part_done(index)` is called once each input has been fully copied, so a
    caller can free its disk space before the next one is read.

    Returns (fields written, fields dropped to keep parity alternating).
    """
    if not specs:
        raise RuntimeError("nothing to merge")

    cut_at = cut_points(specs)

    out_json = None
    all_fields = []
    field_bytes = None
    chroma_out = None
    tbc_out = open(out_base + ".tbc", "wb")
    dropped_parity = 0
    prev_is_first = None

    try:
        for i, spec in enumerate(specs):
            meta = spec["meta"]
            fields = meta["fields"]
            label = spec.get("label") or os.path.basename(spec["tbc"])

            if out_json is None:
                out_json = json.loads(json.dumps(meta))  # keep the caller's copy clean
                field_bytes = field_bytes_of(meta)
                if spec.get("chroma"):
                    chroma_out = open(out_base + "_chroma.tbc", "wb")

            if not fields:
                say("  warning: %s contributed no fields" % label)
                if on_part_done:
                    on_part_done(i)
                continue

            locs = locs_of(spec)
            keep = kept_indices(locs, cut_at[i])
            if not keep:
                say("  warning: %s contributed no fields" % label)
                if on_part_done:
                    on_part_done(i)
                continue

            # Keep first/second field alternating across the join, otherwise
            # every frame after this point pairs the wrong two fields.
            if prev_is_first is not None:
                if bool(fields[keep[0]]["isFirstField"]) == prev_is_first:
                    keep = keep[1:]
                    dropped_parity += 1
                    if not keep:
                        if on_part_done:
                            on_part_done(i)
                        continue

            # Copy field by field rather than slurping the file: one machine's
            # share of a tape is gigabytes, and holding luma and chroma in memory
            # at once would need more RAM than the decode did.
            src, available = _open_checked(spec["tbc"], field_bytes, len(fields),
                                           label, say)
            chroma_src = None
            if chroma_out is not None:
                if not spec.get("chroma") or not os.path.exists(spec["chroma"]):
                    src.close()
                    raise RuntimeError(
                        "%s has no chroma file but an earlier part did; the "
                        "output would be misaligned" % label
                    )
                chroma_src, chroma_available = _open_checked(
                    spec["chroma"], field_bytes, len(fields), label, say)
                # Luma and chroma must stay field-aligned, so a short chroma
                # file limits both.
                available = min(available, chroma_available)
            try:
                for idx in keep:
                    if idx >= available:
                        break
                    src.seek(idx * field_bytes)
                    data = src.read(field_bytes)
                    if len(data) < field_bytes:
                        break
                    tbc_out.write(data)
                    if chroma_src is not None:
                        chroma_src.seek(idx * field_bytes)
                        chroma_out.write(chroma_src.read(field_bytes))
                    entry = dict(fields[idx])
                    entry["fileLoc"] = locs[idx]
                    entry["seqNo"] = len(all_fields) + 1
                    all_fields.append(entry)
                    prev_is_first = bool(fields[idx]["isFirstField"])
            finally:
                src.close()
                if chroma_src is not None:
                    chroma_src.close()

            if on_part_done:
                on_part_done(i)
    finally:
        tbc_out.close()
        if chroma_out is not None:
            chroma_out.close()

    out_json["fields"] = all_fields
    out_json["videoParameters"]["numberOfSequentialFields"] = len(all_fields)
    with open(out_base + ".tbc.json", "w") as fh:
        json.dump(out_json, fh)

    return len(all_fields), dropped_parity
