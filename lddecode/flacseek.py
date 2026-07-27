"""Byte-accurate positioning inside a raw FLAC stream.

An .ldf is a FLAC file whose declared sample rate stands in for the real RF rate
(FLAC cannot express 40 MHz).  ld-compress writes it without a seek table, and on
a long capture the STREAMINFO sample count is wrong as well, so ffmpeg is left
estimating byte positions from the bitrate.  On a 2h45m capture that estimate
came out 25x short: asking for a sample a third of the way in landed near the
start, and the loader then had to decode and discard everything in between.
Correct output, but hours per seek - which makes splitting a capture across
several decoder processes pointless.

FLAC frames carry their own position, so the file is seekable even without an
index: every frame header states its frame number, and for a fixed-blocksize
stream (which is what ld-compress produces) the first sample of frame N is simply
N * blocksize.  This module binary-searches the file by byte offset, resynchronises
on frame headers, and reports where each one starts.

`open_at` then hands back a readable object that looks like a complete FLAC file
starting at the frame containing a given sample: the original metadata headers
followed by the frame data from that offset.  Feeding that to a decoder skips
straight to the right place.
"""

import os
import struct

FRAME_SYNC = 0xFFF8
FRAME_SYNC_MASK = 0xFFFE  # bottom bit of the second byte is the blocking strategy
_SYNC_BYTES = b"\xff\xf8"  # sync word with the fixed-blocksize strategy bit clear

_BLOCKSIZE_TABLE = {
    1: 192,
    2: 576, 3: 1152, 4: 2304, 5: 4608,
    8: 256, 9: 512, 10: 1024, 11: 2048,
    12: 4096, 13: 8192, 14: 16384, 15: 32768,
}

# CRC-8 with polynomial x^8 + x^2 + x + 1, as FLAC uses for frame headers.
_CRC8 = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = ((_c << 1) ^ 0x07) & 0xFF if _c & 0x80 else (_c << 1) & 0xFF
    _CRC8.append(_c)


class NotSeekableFlac(Exception):
    """The file is not a raw FLAC this module can position inside."""


def _crc8(data):
    crc = 0
    for byte in data:
        crc = _CRC8[crc ^ byte]
    return crc


def _read_utf8_number(data, pos):
    """Decode FLAC's UTF-8-style coded number. Returns (value, next_pos)."""
    first = data[pos]
    if first < 0x80:
        return first, pos + 1
    # Count leading ones to get the total length.
    length = 0
    mask = 0x80
    while first & mask:
        length += 1
        mask >>= 1
    if length < 2 or length > 7 or pos + length > len(data):
        return None, pos
    value = first & (0x7F >> length)
    for i in range(1, length):
        byte = data[pos + i]
        if byte & 0xC0 != 0x80:
            return None, pos
        value = (value << 6) | (byte & 0x3F)
    return value, pos + length


class FlacFrameIndex:
    """Locate FLAC frames by sample position, without a seek table."""

    def __init__(self, path):
        self.path = path
        self.size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if fh.read(4) != b"fLaC":
                raise NotSeekableFlac("not a raw FLAC stream (no fLaC magic)")
            self._parse_metadata(fh)

        if self.min_blocksize != self.max_blocksize:
            # Variable-blocksize streams code a sample number rather than a frame
            # number, which is still usable, but ld-compress does not produce them
            # and leaving the path untested would be worse than declining it.
            raise NotSeekableFlac("variable blocksize streams are not supported")
        self.blocksize = self.min_blocksize
        if not self.blocksize:
            raise NotSeekableFlac("STREAMINFO has no blocksize")
        # Only used to seed the very first interpolation step; STREAMINFO may
        # leave it at zero, and an uncompressed frame is the safe upper bound.
        self.max_frame_guess = (self.max_framesize
                                or self.blocksize * self.channels
                                * ((self.bits_per_sample + 7) // 8) + 16)

    def _parse_metadata(self, fh):
        """Read the metadata blocks and remember where the frames start."""
        pos = 4
        while True:
            head = fh.read(4)
            if len(head) < 4:
                raise NotSeekableFlac("truncated metadata")
            is_last = bool(head[0] & 0x80)
            block_type = head[0] & 0x7F
            length = int.from_bytes(head[1:4], "big")
            body = fh.read(length)
            if block_type == 0:
                if length < 34:
                    raise NotSeekableFlac("short STREAMINFO")
                self.min_blocksize, self.max_blocksize = struct.unpack(">HH", body[0:4])
                self.max_framesize = int.from_bytes(body[7:10], "big")
                packed = int.from_bytes(body[10:18], "big")
                self.sample_rate = (packed >> 44) & 0xFFFFF
                self.channels = ((packed >> 41) & 0x07) + 1
                self.bits_per_sample = ((packed >> 36) & 0x1F) + 1
                # Deliberately not trusted: on long captures this field is wrong.
                self.declared_total_samples = packed & 0xFFFFFFFFF
            pos += 4 + length
            if is_last:
                break
        self.first_frame_offset = pos
        fh.seek(0)
        self.header_bytes = fh.read(pos)

    # -- frame header parsing -------------------------------------------------

    def _parse_frame_header(self, buf, pos):
        """Validate a frame header at buf[pos:]. Returns its first sample, or None."""
        if pos + 16 > len(buf):
            return None
        if (int.from_bytes(buf[pos:pos + 2], "big") & FRAME_SYNC_MASK) != FRAME_SYNC:
            return None

        blocking_strategy = buf[pos + 1] & 0x01
        if blocking_strategy:
            # Variable blocksize; we rejected those in __init__.
            return None

        blocksize_bits = buf[pos + 2] >> 4
        samplerate_bits = buf[pos + 2] & 0x0F
        channel_bits = buf[pos + 3] >> 4
        samplesize_bits = (buf[pos + 3] >> 1) & 0x07
        if buf[pos + 3] & 0x01:
            return None  # reserved bit must be zero
        if blocksize_bits == 0 or samplerate_bits == 0x0F:
            return None

        # Cross-check against STREAMINFO: a random byte pattern that happens to
        # look like a sync word almost never agrees on all of these.
        if blocksize_bits in _BLOCKSIZE_TABLE:
            if _BLOCKSIZE_TABLE[blocksize_bits] != self.blocksize:
                return None
        elif blocksize_bits not in (6, 7):
            return None
        if channel_bits + 1 != self.channels and channel_bits < 8:
            return None

        number, after = _read_utf8_number(buf, pos + 4)
        if number is None:
            return None
        # Optional blocksize / sample rate fields sit between the coded number
        # and the CRC.
        if blocksize_bits == 6:
            after += 1
        elif blocksize_bits == 7:
            after += 2
        if samplerate_bits == 12:
            after += 1
        elif samplerate_bits in (13, 14):
            after += 2
        if after >= len(buf):
            return None
        if _crc8(buf[pos:after]) != buf[after]:
            return None

        return number * self.blocksize

    def _scan_forward(self, fh, byte_pos, window=1 << 16, limit=1 << 22, stop=None):
        """First valid frame at or after `byte_pos`. Returns (offset, first_sample).

        `stop` bounds the search so a binary step cannot run past its bracket.
        """
        byte_pos = max(byte_pos, self.first_frame_offset)
        end = self.size if stop is None else min(stop, self.size)
        searched = 0
        while byte_pos < end and searched < limit:
            fh.seek(byte_pos)
            buf = fh.read(window)
            if len(buf) < 16:
                return None
            # bytes.find does the byte hunting in C; stepping through the window
            # in Python was costing more than the disk reads by a wide margin.
            # A fixed-blocksize frame always starts 0xFFF8, so this rejects all
            # but roughly one position in 65536 without any Python per byte.
            limit_i = len(buf) - 16  # leave room for the longest header
            i = buf.find(_SYNC_BYTES)
            while 0 <= i < limit_i:
                sample = self._parse_frame_header(buf, i)
                if sample is not None and self._confirms(fh, byte_pos + i, sample):
                    # A lone valid-looking header can still be a fluke inside
                    # encoded audio, hence the confirmation above.
                    return byte_pos + i, sample
                i = buf.find(_SYNC_BYTES, i + 1)
            byte_pos += window - 16
            searched += window - 16
        return None

    def _confirms(self, fh, offset, sample):
        """Is there a frame right after `offset` whose number follows on?"""
        span = self.max_frame_guess * 2 + 64
        fh.seek(offset)
        buf = fh.read(span)
        i = buf.find(_SYNC_BYTES, 1)
        while 0 <= i < len(buf) - 16:
            nxt = self._parse_frame_header(buf, i)
            if nxt is not None:
                return nxt == sample + self.blocksize
            i = buf.find(_SYNC_BYTES, i + 1)
        # Ran out of file: the last frame has nothing to confirm it.
        return offset + span >= self.size

    # -- public API -----------------------------------------------------------

    def find_frame(self, target_sample):
        """Byte offset and first sample of the frame containing `target_sample`.

        `target_sample` and the returned sample are both counted from the start
        of this file, not from the frame numbers written in it.  That matters for
        a file cut out of a longer capture: its frames keep the numbering they had
        in the original, so frame numbers there start at some large value.

        A plain bisection on byte offset.  The bracket is (lo, hi], where the
        frame at `lo` starts at or before the target and nothing between `lo` and
        `hi` has been ruled in yet.  Frames are near enough a constant size that
        interpolating instead of halving lands within a few frames on the first
        step, so even a few hundred GB takes a handful of reads.
        """
        # Close enough that walking the remaining frames beats another bisection.
        settle = 1 << 16

        with open(self.path, "rb") as fh:
            first = self._scan_forward(fh, self.first_frame_offset)
            if first is None:
                raise NotSeekableFlac("no frame header found at the start of the file")
            # Everything below works in the file's own frame numbering; the
            # caller's offsets are relative to the first frame in this file.
            self.base_sample = first[1]
            target_sample += first[1]
            if target_sample <= first[1]:
                return first[0], 0

            lo_off, lo_sample = first
            hi_off = self.size
            hi_sample = None  # sample of the first frame at or after hi_off

            # Interpolation alone can stall: when the estimated rate is very
            # close to the truth the guess lands a byte below the bound, finds
            # the same frame again, and the bracket shrinks by one byte per
            # step.  Forcing a plain bisection every other iteration bounds the
            # worst case at ~2*log2(size) reads while keeping interpolation's
            # near-instant convergence in the normal case.
            bisect_turn = False

            while hi_off - lo_off > settle:
                span_bytes = hi_off - lo_off
                if bisect_turn:
                    guess = lo_off + span_bytes // 2
                else:
                    guess = lo_off + int(span_bytes * self._fraction(
                        target_sample, lo_sample, hi_sample, span_bytes, first, lo_off
                    ))
                bisect_turn = not bisect_turn
                # Keep the step strictly inside the bracket so it always shrinks.
                guess = min(max(guess, lo_off + 1), hi_off - 1)

                found = self._scan_forward(fh, guess, stop=hi_off)
                if found is None:
                    # No frame between the guess and the top of the bracket.
                    hi_off, hi_sample = guess, hi_sample
                    continue

                off, sample = found
                if sample <= target_sample:
                    lo_off, lo_sample = off, sample
                else:
                    # `off` is the *first* frame at or after the guess, and it is
                    # already past the target, so nothing from the guess onwards
                    # can qualify.  Pulling the bound back to the guess rather
                    # than to `off` also guarantees the bracket shrinks - `off`
                    # can equal the current bound, which would loop forever.
                    hi_off, hi_sample = guess, sample

            # Walk the last stretch frame by frame for an exact answer.
            best = (lo_off, lo_sample)
            probe = lo_off
            while True:
                found = self._scan_forward(fh, probe + 1, stop=hi_off + settle)
                if found is None or found[1] > target_sample:
                    break
                best = found
                probe = found[0]
            return best[0], best[1] - first[1]

    def _fraction(self, target, lo_sample, hi_sample, span_bytes, first, lo_off):
        """Where in the current bracket the target probably sits, as 0..1.

        Uses whichever samples-per-byte rate is best established: the two ends of
        the bracket if both are known, otherwise the average from the start of
        the file.  STREAMINFO's sample count is never used - it is exactly the
        field that is wrong on long captures.
        """
        if hi_sample is not None and hi_sample > lo_sample:
            frac = (target - lo_sample) / float(hi_sample - lo_sample)
        else:
            seen_bytes = lo_off - first[0]
            seen_samples = lo_sample - first[1]
            if seen_bytes > 0 and seen_samples > 0:
                rate = seen_samples / float(seen_bytes)
            else:
                # Nothing measured yet: a fully packed frame is the densest the
                # file can be, so this errs towards guessing too far in, which
                # establishes an upper bound on the first step.
                rate = self.blocksize / float(self.max_frame_guess)
            span_samples = span_bytes * rate
            frac = (target - lo_sample) / span_samples if span_samples > 0 else 0.5
        return min(max(frac, 0.0), 1.0)

    def open_at(self, target_sample):
        """A file-like FLAC stream that starts at the frame holding `target_sample`.

        Returns (stream, first_sample) - the decoder will produce `first_sample`
        first, so the caller discards `target_sample - first_sample` samples.
        """
        offset, first_sample = self.find_frame(target_sample)
        return _SplicedFlac(self.path, self.header_bytes, offset), first_sample


class _SplicedFlac:
    """The original FLAC headers followed by frame data from a byte offset.

    A decoder reading this sees a well-formed FLAC file that happens to begin
    part-way through the recording.
    """

    def __init__(self, path, header_bytes, offset):
        self._header = header_bytes
        self._header_pos = 0
        self._fh = open(path, "rb")
        self._fh.seek(offset)

    def read(self, size=-1):
        if size is None or size < 0:
            size = 1 << 20
        out = b""
        if self._header_pos < len(self._header):
            out = self._header[self._header_pos:self._header_pos + size]
            self._header_pos += len(out)
            size -= len(out)
            if not size:
                return out
        return out + self._fh.read(size)

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
