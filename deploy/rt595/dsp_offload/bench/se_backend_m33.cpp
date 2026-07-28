/*
 * se_backend_m33.cpp — the "without HiFi4" backend: local int8 TFLite-Micro on the M33,
 * using the CMSIS-NN kernels. Just wraps the existing SE_* driver
 * (../../app/model_se_stream.cpp + model_ops_micro.cpp, built with the int8 model headers)
 * into the se_backend_t vtable. No new inference code.
 *
 * last_compute_cycles()=0 and compute_clock_hz()=0 tell the bench "the M33-measured
 * latency IS the compute cost" for this backend.
 */
#include "se_backend.h"

static uint32_t m33_zero(void) { return 0; }

extern "C" const se_backend_t se_backend_m33 = {
    /* name                */ "M33 int8 (CMSIS-NN)",
    /* init                */ SE_Init,
    /* reset               */ SE_ResetStates,
    /* process             */ SE_ProcessFrame,
    /* arena_used          */ SE_ArenaUsedBytes,
    /* last_compute_cycles */ m33_zero,
    /* compute_clock_hz    */ m33_zero,
};
