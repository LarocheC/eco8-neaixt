/*
 * se_backend_hifi4.cpp — the "with HiFi4" backend: forwards each frame to the DSP via the
 * RPC core (se_dsp_*, ../cm33/se_remote.cpp). last_compute_cycles() returns the DSP's own
 * CCOUNT for the last frame (pure kernel time), and compute_clock_hz() the DSP clock — so
 * the bench can separate raw HiFi4 compute from the M33-observed round-trip latency.
 */
#include "se_backend.h"

extern "C" const se_backend_t se_backend_hifi4 = {
    /* name                */ "HiFi4 int8 (xa_nnlib)",
    /* init                */ se_dsp_init,
    /* reset               */ se_dsp_reset,
    /* process             */ se_dsp_process,
    /* arena_used          */ se_dsp_arena,
    /* last_compute_cycles */ se_dsp_last_cycles,
    /* compute_clock_hz    */ se_dsp_clock_hz,
};
