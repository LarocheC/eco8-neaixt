/*
 * LiSenNet streaming speech-enhancement driver for TFLite-Micro on the RT595 M33.
 *
 * The RT595 analog of deploy/stm32n6/app/ai_dpu_se_stream.c, but on the TFLM
 * MicroInterpreter instead of the ST STAI runtime. One MicroInterpreter runs the
 * frame-by-frame LiSenNet graph: a 3-channel feature frame in, an enhanced
 * compressed-magnitude frame out, with the K recurrent state tensors carried across
 * Invoke()s. All I/O positions, quant params and the state-feedback map come from the
 * auto-generated model_io_layout.h, so this driver is model-agnostic.
 */
#ifndef MODEL_SE_STREAM_H
#define MODEL_SE_STREAM_H

#include <stdint.h>
#include "fsl_common.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* Build the interpreter, allocate tensors, zero the recurrent state. */
status_t SE_Init(void);

/* Zero all recurrent state (call between independent utterances). */
void SE_ResetStates(void);

/* Run one frame.
 *   feat_in : 3 * MODEL_FEATURE_LEN floats — the 3-channel network feature
 *             (compressed mag, group-delay/pi, IFD/pi), channel-major [3][F].
 *   mask_out: MODEL_FEATURE_LEN floats — enhanced compressed magnitude for the frame.
 * State feedback (output states -> input states) is handled internally. */
status_t SE_ProcessFrame(const float *feat_in, float *mask_out);

/* Tensor-arena bytes actually used (valid after SE_Init) — for sizing reports. */
uint32_t SE_ArenaUsedBytes(void);

#if defined(__cplusplus)
}
#endif

#endif /* MODEL_SE_STREAM_H */
