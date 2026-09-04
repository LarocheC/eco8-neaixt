/*
 * se_backend.h — a tiny vtable so the bench can drive M33-local and HiFi4-offloaded
 * inference through one interface and print them side by side.
 *
 * Also declares the DSP RPC primitives (implemented in ../cm33/se_remote.cpp) and the
 * public SE_* API (so ../cm33/se_remote_alias.c can alias it in the production build).
 */
#ifndef SE_BACKEND_H
#define SE_BACKEND_H

#include <stdint.h>
#include "fsl_common.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* One speech-enhancement backend. process() runs a single streaming frame:
 *   feat[3*MODEL_FEATURE_LEN] floats -> mask[MODEL_FEATURE_LEN] floats.
 * last_compute_cycles() returns the backend's own view of the last frame's COMPUTE cost
 * (DSP CCOUNT for HiFi4; 0 for M33 — there the M33-measured latency IS the compute). */
typedef struct {
    const char *name;
    status_t (*init)(void);
    void     (*reset)(void);
    status_t (*process)(const float *feat, float *mask);
    uint32_t (*arena_used)(void);
    uint32_t (*last_compute_cycles)(void);   /* 0 if same as the M33-measured latency */
    uint32_t (*compute_clock_hz)(void);       /* clock for last_compute_cycles(); 0 => M33 clock */
} se_backend_t;

extern const se_backend_t se_backend_m33;     /* local int8 TFLM (CMSIS-NN) — "without HiFi4" */
extern const se_backend_t se_backend_hifi4;   /* DSP offload (xa_nnlib)     — "with HiFi4"    */

/* DSP RPC primitives (../cm33/se_remote.cpp). */
status_t se_dsp_init(void);
void     se_dsp_reset(void);
status_t se_dsp_process(const float *feat, float *mask);
uint32_t se_dsp_arena(void);
uint32_t se_dsp_last_cycles(void);
uint32_t se_dsp_clock_hz(void);

/* Public SE_* API (../../app/model_se_stream.h). Declared here so se_remote_alias.c can
 * alias it; the M33-local implementation is ../../app/model_se_stream.cpp. */
status_t SE_Init(void);
void     SE_ResetStates(void);
status_t SE_ProcessFrame(const float *feat, float *mask);
uint32_t SE_ArenaUsedBytes(void);

#if defined(__cplusplus)
}
#endif

#endif /* SE_BACKEND_H */
