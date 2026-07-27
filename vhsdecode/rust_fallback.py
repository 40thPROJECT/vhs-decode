"""Pure numpy/numba equivalents of the `vhsd_rust` extension.

The rust extension is optional at build time (it needs a cargo toolchain), but
`vhsdecode.demod` and `vhsdecode.main` imported it unconditionally, so a source
checkout without it could not run at all.  This module provides drop-in
replacements so the decoder works either way; when `vhsd_rust` is importable it
is still preferred, since the hand-vectorised rust code is faster.

The implementations here deliberately mirror the rust ones bit-for-bit in
behaviour (not in floating point rounding - rust uses a polynomial atan2
approximation on f32, we use the exact libm atan2 on f64).
"""

import numpy as np
from numba import njit

TAU = 2.0 * np.pi


@njit(cache=True, nogil=True, fastmath=True)
def _unwrap_hilbert_nb(hilbert, freq_hz):
    """Instantaneous frequency of an analytic signal, in Hz.

    Mirrors `hilbert_all` in src/lib.rs: for each pair of consecutive samples
    take the phase difference wrapped into [0, 2pi) and scale it to Hz.  Element
    0 has no predecessor and is left at zero, same as the rust version.
    """
    out = np.zeros(len(hilbert), dtype=np.float64)
    prev = np.arctan2(hilbert[0].imag, hilbert[0].real)
    for i in range(1, len(hilbert)):
        cur = np.arctan2(hilbert[i].imag, hilbert[i].real)
        diff = cur - prev
        # wrap into [0, TAU)
        diff = diff - np.floor(diff / TAU) * TAU
        out[i] = diff * freq_hz / TAU
        prev = cur
    return out


def unwrap_hilbert(hilbert, freq_hz):
    return _unwrap_hilbert_nb(np.ascontiguousarray(hilbert), float(freq_hz))


def complex_angle_py(input_array):
    return np.angle(input_array)


def unwrap_angles(input_array):
    return np.unwrap(input_array)


def diff_forward_in_place(input_array):
    """np.ediff1d(input_array, to_begin=0) applied in place."""
    input_array[1:] -= input_array[:-1].copy()
    input_array[0] = 0.0


def check_debug():
    """The rust extension reports whether it was built with debug assertions.

    There is no rust build here, so there is nothing to warn about.
    """
    return False
