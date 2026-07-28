/*
 * se_remote_alias.c — production HiFi4-only build: map the public SE_* API onto the DSP
 * RPC (se_dsp_*, in se_remote.cpp). Linking this makes se_remote.cpp a true drop-in for
 * ../../app/model_se_stream.cpp, so ../../app/main.cpp is unchanged.
 *
 * DO NOT link this in the bench build — there the M33-local model_se_stream.cpp already
 * provides SE_* (the "without HiFi4" backend), and the bench reaches the DSP through the
 * se_backend_t vtable instead. Linking both here would collide on SE_*.
 */
#include "se_backend.h"

status_t SE_Init(void)                                    { return se_dsp_init(); }
void     SE_ResetStates(void)                             { se_dsp_reset(); }
status_t SE_ProcessFrame(const float *feat, float *mask)  { return se_dsp_process(feat, mask); }
uint32_t SE_ArenaUsedBytes(void)                          { return se_dsp_arena(); }
