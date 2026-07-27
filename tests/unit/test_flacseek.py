"""Positioning inside a raw FLAC by frame header rather than by container timestamp."""

import os
import unittest

import numpy as np

from lddecode.flacseek import FlacFrameIndex, NotSeekableFlac

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CAPTURE = os.path.abspath(os.path.join(DATA, "vhs_pal.flac"))


def decode_all(path):
    """Every sample in the file, decoded straight through."""
    import av

    out = bytearray()
    with av.open(path) as container:
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono")
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                out += bytes(resampled.planes[0])
    return np.frombuffer(bytes(out), "<i2")


def decode_from(index, target, count):
    """`count` samples starting at `target`, reached by splicing."""
    import av

    stream, first_sample = index.open_at(target)
    skip = target - first_sample
    out = bytearray()
    try:
        with av.open(stream) as container:
            resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono")
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    data = bytes(resampled.planes[0])
                    if skip:
                        drop = min(skip * 2, len(data))
                        data = data[drop:]
                        skip -= drop // 2
                    out += data
                    if len(out) >= count * 2:
                        return np.frombuffer(bytes(out[: count * 2]), "<i2")
    finally:
        stream.close()
    return np.frombuffer(bytes(out), "<i2")


@unittest.skipUnless(os.path.exists(CAPTURE), "test data submodule not checked out")
class TestFlacFrameIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = FlacFrameIndex(CAPTURE)
        cls.samples = decode_all(CAPTURE)

    def test_reads_streaminfo(self):
        self.assertGreater(self.index.blocksize, 0)
        self.assertEqual(self.index.min_blocksize, self.index.max_blocksize)
        self.assertGreater(self.index.first_frame_offset, 4)

    def test_lands_on_the_frame_holding_the_target(self):
        total = len(self.samples)
        for target in (0, 1, total // 7, total // 3, total // 2, total - 5000):
            with self.subTest(target=target):
                offset, first = self.index.find_frame(target)
                self.assertLessEqual(first, target, "landed after the target")
                self.assertLess(target - first, self.index.blocksize,
                                "landed more than one frame early")
                self.assertGreaterEqual(offset, self.index.first_frame_offset)
                self.assertLess(offset, self.index.size)

    def test_spliced_stream_decodes_the_right_samples(self):
        total = len(self.samples)
        count = 2048
        for target in (0, 12345, total // 3, total // 2, total - count - 1):
            with self.subTest(target=target):
                got = decode_from(self.index, target, count)
                want = self.samples[target:target + count]
                self.assertEqual(len(got), len(want))
                np.testing.assert_array_equal(got, want)

    def test_declines_what_it_cannot_position(self):
        # Anything that is not a raw FLAC has to be refused rather than
        # mispositioned, so callers can fall back to the container's own seek.
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".ldf", delete=False) as fh:
            fh.write(b"OggS" + b"\0" * 4096)
            path = fh.name
        try:
            with self.assertRaises(NotSeekableFlac):
                FlacFrameIndex(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
