"""Inserting a decode into a gap in another decode, in place."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

W, H = 20, 4
FIELD_BYTES = W * H * 2
SPF = 1000                      # samples per field
FS = 40e6
FPS = FS / (SPF * 2)            # makes the tool's samples-per-field come out as SPF


def tag(loc):
    """A byte pattern identifying the field at this position on the tape."""
    return ((loc // SPF) % 65536).to_bytes(2, "little") * (W * H)


def write_tbc(base, locs, chroma=True):
    fields = []
    with open(base + ".tbc", "wb") as fh:
        for n, loc in enumerate(locs):
            fields.append({"seqNo": n + 1, "fileLoc": loc,
                           "isFirstField": (loc // SPF) % 2 == 0})
            fh.write(tag(loc))
    if chroma:
        with open(base + "_chroma.tbc", "wb") as fh:
            for loc in locs:
                fh.write(tag(loc))
    with open(base + ".tbc.json", "w") as fh:
        json.dump({"videoParameters": {"fieldWidth": W, "fieldHeight": H,
                                       "numberOfSequentialFields": len(fields)},
                   "fields": fields}, fh)


class TestInsertTbc(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.into = os.path.join(self.tmp, "into")
        self.ins = os.path.join(self.tmp, "ins")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_tool(self):
        return subprocess.run(
            [sys.executable, "insert_tbc.py", "--into", self.into + ".tbc",
             "--insert", self.ins + ".tbc", "--fps", str(FPS)],
            cwd=REPO, capture_output=True, text=True)

    def result(self):
        meta = json.load(open(self.into + ".tbc.json"))
        data = open(self.into + ".tbc", "rb").read()
        chroma = (open(self.into + "_chroma.tbc", "rb").read()
                  if os.path.exists(self.into + "_chroma.tbc") else None)
        return meta, data, chroma

    def check_filled(self, head, tail, insert, chroma=True):
        write_tbc(self.into, head + tail, chroma)
        write_tbc(self.ins, insert, chroma)
        before = os.path.getsize(self.into + ".tbc")

        proc = self.run_tool()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

        meta, data, ch = self.result()
        locs = [f["fileLoc"] for f in meta["fields"]]
        expected = sorted(head + tail + [l for l in insert if head[-1] < l < tail[0]])

        self.assertEqual(locs, expected, "wrong fields after the insert")
        self.assertEqual(len(locs), len(set(locs)), "duplicated a field")
        self.assertEqual(data, b"".join(tag(l) for l in locs),
                         "pixels do not match the positions claimed")
        if ch is not None:
            self.assertEqual(ch, data, "chroma drifted out of step with luma")
        self.assertEqual(len(data), len(locs) * FIELD_BYTES)
        self.assertEqual(meta["videoParameters"]["numberOfSequentialFields"],
                         len(locs))
        self.assertEqual([f["seqNo"] for f in meta["fields"]],
                         list(range(1, len(locs) + 1)))
        parity = [bool(f["isFirstField"]) for f in meta["fields"]]
        self.assertTrue(all(a != b for a, b in zip(parity, parity[1:])),
                        "field parity stopped alternating at a join")
        self.assertEqual(len(data) - before,
                         (len(locs) - len(head) - len(tail)) * FIELD_BYTES,
                         "file did not grow by exactly the inserted fields")

    HEAD = [0, 1000, 2000, 3000, 4000, 5000]
    TAIL = [11000, 12000, 13000, 14000]

    def test_overrun_on_both_sides_is_trimmed(self):
        self.check_filled(self.HEAD, self.TAIL,
                          [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
                           11000, 12000])

    def test_insert_just_covers_the_gap(self):
        self.check_filled(self.HEAD, self.TAIL, [6000, 7000, 8000, 9000, 10000])

    def test_insert_covers_only_part_of_the_gap(self):
        self.check_filled(self.HEAD, self.TAIL, [6000, 7000, 8000])

    def test_without_chroma(self):
        self.check_filled(self.HEAD, self.TAIL, [6000, 7000, 8000, 9000, 10000],
                          chroma=False)

    def test_long_tail_shifts_correctly(self):
        # The tail is moved backwards in chunks; this spans many of them.
        self.check_filled(self.HEAD, [11000 + i * 1000 for i in range(400)],
                          [6000, 7000, 8000, 9000, 10000])

    def test_refuses_when_there_is_no_gap(self):
        write_tbc(self.into, [0, 1000, 2000, 3000])
        write_tbc(self.ins, [6000, 7000])
        proc = self.run_tool()
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("no gap", proc.stdout + proc.stderr)

    def test_refuses_an_insert_that_misses_the_gap(self):
        write_tbc(self.into, [0, 1000, 2000, 11000, 12000])
        write_tbc(self.ins, [50000, 51000])
        proc = self.run_tool()
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("do not overlap", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
