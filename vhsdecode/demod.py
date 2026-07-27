from numba import njit
import numpy as np

try:
    import vhsd_rust

    _unwrap_hilbert_impl = vhsd_rust.unwrap_hilbert
except ModuleNotFoundError:
    # No cargo toolchain / rust extension built - use the numba equivalent.
    from vhsdecode.rust_fallback import unwrap_hilbert as _unwrap_hilbert_impl


@njit(cache=True, nogil=True)
def replace_spikes(demod, demod_diffed, max_value, replace_start=8, replace_end=30):
    """Go through and replace spikes and some samples after them with data
    from the diff demod pass"""
    assert len(demod) == len(
        demod_diffed
    ), "diff demod length doesn't match demod length"
    too_high = max_value
    to_fix = np.where(demod > too_high)[0]

    for i in to_fix:
        start = max(i - replace_start, 0)
        end = min(i + replace_end, len(demod_diffed) - 1)
        # Only replace if it seems to help
        if max(demod_diffed[start:end]) < max(demod[start:end]):
            demod[start:end] = demod_diffed[start:end]

    return demod


@njit(cache=True, nogil=True)
def smooth_spikes(demod, max_value):
    """Go through spikes above max value and replace with the average of the neighbours."""
    too_high = max_value
    # Note - optimization, avoid first/last value so we don't have to check
    # array bounds later.
    to_fix = np.where(demod[1:-1] > too_high)[0]

    for i in to_fix:
        demod[i + 1] = (demod[i] + demod[i + 2]) / 2

    return demod


def unwrap_hilbert(hilbert, freq_hz):
    # return hilbert_test.unwrap_hilbert(hilbert, freq_hz)
    return _unwrap_hilbert_impl(hilbert, freq_hz)
