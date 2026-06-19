/**
 * ai_dpu_se_stream.h — streaming speech-enhancement DPU glue for a multi-input/
 * multi-output ConvFSENet model with external int8 recurrent state, on STM32N6.
 *
 * Replaces the single-input/single-output assumption baked into the stock
 * STM32N6-GettingStarted-Audio Projects/Dpu/ai_dpu.c (which hard-asserts
 * n_inputs==1 && n_outputs==1). Here the NPU model has:
 *   input  [0]            = noisy magnitude spectrum, float32, 257 bins
 *   input  [1..N_STATES]  = int8 recurrent FIFO state (fed back from prev frame)
 *   output [0]            = enhancement mask, int8, 257 bins  -> dequantized to [0,1]
 *   output [1..N_STATES]  = int8 next-frame state            -> copied back to inputs
 *
 * Per-frame contract: caller supplies the float magnitude, gets a float mask.
 * The STFT/iSTFT stay on the Cortex-M55 (CMSIS-DSP) — this module only owns the
 * NPU inference + state plumbing + int8<->float boundary.
 */
#ifndef AI_DPU_SE_STREAM_H
#define AI_DPU_SE_STREAM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Initialize the runtime + network, bind I/O buffers, zero the recurrent state.
 *  Returns 0 on success, negative STAI error code otherwise. Call once at boot. */
int  se_stream_init(void);

/** Run one frame: dequantized mask[257] out, given magnitude[257] in.
 *  Advances the recurrent state internally. Returns 0 on success. */
int  se_stream_process_frame(const float *mag_in, float *mask_out);

/** Reset recurrent state to silence (int8 code for 0.0). Call on stream restart. */
void se_stream_reset_state(void);

#ifdef __cplusplus
}
#endif

#endif /* AI_DPU_SE_STREAM_H */
