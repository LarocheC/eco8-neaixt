/*
 * LiSenNet streaming SE driver — TFLite-Micro on the RT595 Cortex-M33.
 * See model_se_stream.h. Interpreter setup follows the mcuxsdk eiq common/tflm/model.cpp
 * pattern; the multi-I/O + recurrent-state logic is specific to the streaming graph.
 */
#include <string.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "fsl_debug_console.h"
#include "model_se_stream.h"
#include "model_data.h"        /* const unsigned char model_data[]; model_data_len */
#include "model_io_layout.h"   /* MODEL_* + MODEL_STATE_MAP (auto-generated) */

extern "C" tflite::MicroOpResolver &MODEL_GetOpsResolver(void);   /* model_ops_micro.cpp */

/* Tensor arena. Override -DSE_TENSOR_ARENA_SIZE=... after reading the "used" print
 * from SE_Init(). Placed in ordinary RAM (M33, no NPU cache-coherency concern). */
#ifndef SE_TENSOR_ARENA_SIZE
#define SE_TENSOR_ARENA_SIZE (512 * 1024)   /* fp32 conv-hardened needs more; tune from the
                                               AllocateTensors "used" print on first boot. */
#endif
static uint8_t s_arena[SE_TENSOR_ARENA_SIZE] __attribute__((aligned(16)));

static tflite::MicroInterpreter *s_interp = nullptr;
static uint32_t s_arenaUsed = 0;

static inline int8_t quant_i8(float x, float scale, int zp)
{
    int32_t q = (int32_t)lrintf(x / scale) + zp;
    if (q < -128) q = -128;
    if (q >  127) q =  127;
    return (int8_t)q;
}

/* Reject an export whose recurrent state is numerically dead.
 *
 * ai-edge-quantizer gives state_in and state_out independent scales. When the
 * calibration never exercised a state port, its input scale collapses onto the
 * calibrator floor (~3.92e-12 in practice). An int8 tensor with that scale spans
 * about +/-5e-10, so EVERY real state value saturates to one code: the recurrence
 * silently stops carrying information and the model runs as if it were stateless.
 * It still executes, still produces plausible output, and still benchmarks at
 * exactly the same cycle count -- the loop is data-independent -- so nothing
 * downstream notices. Observed on ConvFSENet (9/9 states) and NSNet2 (1/1);
 * LiSenNet relu6-deep is clean (0/25).
 *
 * The driver cannot repair a garbage scale, so it refuses to run on one. Define
 * SE_ALLOW_DEAD_STATE=1 to measure cycles anyway -- the timing is still valid,
 * the enhancement is not. */
/* Max Q16 ratio the 32-bit requant loop can evaluate without overflow.
 *
 * The loop computes (d * ratio) + 32768 in int32, where d = sout - out_zp is
 * bounded by +/-255 (both are int8). So ratio must satisfy
 *     255 * ratio + 32768 <= INT32_MAX   ->   ratio <= 8,421,313
 * We use 8,000,000 (a Q16 gain of ~122x) for margin. This is not a limitation in
 * practice: a healthy recurrence has in_scale ~= out_scale, i.e. a gain near 1.0
 * (ratio ~65536). A gain of 122x already means the state input can only represent
 * 1/122 of the range the state output produces -- pathological well before the
 * arithmetic limit. Validating this ONCE in SE_CheckStateScales is what lets the
 * per-element loop stay 32-bit; making the loop 64-bit instead cost +35% on
 * LiSenNet (8,902,092 -> 12,002,304 cyc/frame), which is not a price worth paying
 * to keep undefined behaviour well-defined. */
#ifndef SE_STATE_RATIO_MAX
#define SE_STATE_RATIO_MAX (8000000.0)
#endif

static status_t SE_CheckStateScales(void)
{
    int dead = 0;
    for (int k = 0; k < MODEL_N_STATES; k++)
    {
        const model_state_map_t *m = &MODEL_STATE_MAP[k];
        if (m->memcpy_safe) continue;
        if (!(m->in_scale > 0.0f) || !(m->out_scale > 0.0f))
        {
            PRINTF("state %d: non-positive scale (in %g out %g)\r\n",
                   k, (double)m->in_scale, (double)m->out_scale);
            dead++;
            continue;
        }
        double ratio = (double)m->out_scale / (double)m->in_scale * 65536.0;
        if (ratio > SE_STATE_RATIO_MAX)
        {
            PRINTF("state %2d: in_scale %g is collapsed vs out_scale %g (Q16 ratio %.3g)"
                   " -- every state code saturates\r\n",
                   k, (double)m->in_scale, (double)m->out_scale, ratio);
            dead++;
        }
    }
    if (dead == 0) return kStatus_Success;

    PRINTF("\r\n*** %d/%d recurrent states are numerically DEAD ***\r\n"
           "The int8 export's state input scales collapsed onto the calibration floor, so\r\n"
           "the recurrence carries no information -- this model runs as if it were stateless.\r\n"
           "Fix the export (calibrate with real propagated state), not the firmware.\r\n",
           dead, (int)MODEL_N_STATES);
#if SE_ALLOW_DEAD_STATE
    PRINTF("SE_ALLOW_DEAD_STATE set: continuing. CYCLES ARE VALID, OUTPUT IS NOT.\r\n\r\n");
    return kStatus_Success;
#else
    PRINTF("Refusing to run. Rebuild with -DSE_ALLOW_DEAD_STATE=1 to measure cycles anyway.\r\n\r\n");
    return kStatus_Fail;
#endif
}

status_t SE_Init(void)
{
    const tflite::Model *model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        PRINTF("Model schema %d != supported %d\r\n", model->version(), TFLITE_SCHEMA_VERSION);
        return kStatus_Fail;
    }

    static tflite::MicroInterpreter interp(model, MODEL_GetOpsResolver(),
                                           s_arena, SE_TENSOR_ARENA_SIZE);
    s_interp = &interp;

    if (s_interp->AllocateTensors() != kTfLiteOk)
    {
        PRINTF("AllocateTensors() failed (arena too small? have %u B)\r\n",
               (unsigned)SE_TENSOR_ARENA_SIZE);
        return kStatus_Fail;
    }
    s_arenaUsed = (uint32_t)s_interp->arena_used_bytes();

    if (s_interp->inputs_size() != MODEL_N_IO || s_interp->outputs_size() != MODEL_N_IO)
    {
        PRINTF("IO count drift: model %u/%u vs header %u\r\n",
               (unsigned)s_interp->inputs_size(), (unsigned)s_interp->outputs_size(),
               (unsigned)MODEL_N_IO);
        return kStatus_Fail;
    }

    PRINTF("SE model: %u IO, %u states; arena used %u / %u B\r\n",
           (unsigned)MODEL_N_IO, (unsigned)MODEL_N_STATES,
           (unsigned)s_arenaUsed, (unsigned)SE_TENSOR_ARENA_SIZE);

    if (SE_CheckStateScales() != kStatus_Success) return kStatus_Fail;

    SE_ResetStates();
    return kStatus_Success;
}

void SE_ResetStates(void)
{
    /* Zero float state == the input zero-point code for each state tensor. */
    for (int k = 0; k < MODEL_N_STATES; k++)
    {
        const model_state_map_t *m = &MODEL_STATE_MAP[k];
        int8_t *in = s_interp->input(m->in_pos)->data.int8;
        memset(in, (int8_t)m->in_zp, (size_t)m->count);
    }
}

status_t SE_ProcessFrame(const float *feat_in, float *mask_out)
{
    /* 1. Feed the feature into the feat input tensor. int8 model: quantise;
     *    fp32 model: copy floats straight through. (Gated at compile time per model.)
     *    MODEL_FEATURE_TOTAL is the product of ALL feat dims — 3*F for LiSenNet's
     *    3-channel feature, but 1*F for ConvFSENet/NSNet2. The old hardcoded `3 *`
     *    over-fed those models by 3x. */
#ifndef MODEL_FEATURE_TOTAL
#define MODEL_FEATURE_TOTAL (3 * MODEL_FEATURE_LEN)   /* pre-multi-model header fallback */
#endif
    const int feat_n = MODEL_FEATURE_TOTAL;
#if MODEL_FEATURE_IS_INT8
    int8_t *feat = s_interp->input(MODEL_FEATURE_IN_POS)->data.int8;
    for (int i = 0; i < feat_n; i++)
        feat[i] = quant_i8(feat_in[i], MODEL_FEATURE_SCALE, MODEL_FEATURE_ZEROPOINT);
#else
    float *feat = s_interp->input(MODEL_FEATURE_IN_POS)->data.f;
    for (int i = 0; i < feat_n; i++) feat[i] = feat_in[i];
#endif

    /* 2. Run. State inputs already hold the previous frame's propagated state. */
    if (s_interp->Invoke() != kTfLiteOk)
    {
        PRINTF("Invoke failed\r\n");
        return kStatus_Fail;
    }

    /* 3. Read the enhanced-magnitude output (dequantise int8, copy fp32). */
#if MODEL_FEATURE_IS_INT8
    const int8_t *mask = s_interp->output(MODEL_MASK_OUT_POS)->data.int8;
    for (int i = 0; i < MODEL_MASK_LEN; i++)
        mask_out[i] = ((int)mask[i] - MODEL_MASK_ZEROPOINT) * MODEL_MASK_SCALE;
#else
    const float *mask = s_interp->output(MODEL_MASK_OUT_POS)->data.f;
    for (int i = 0; i < MODEL_MASK_LEN; i++) mask_out[i] = mask[i];
#endif

    /* 4. Feed state outputs back to the matching state inputs for the next frame.
     *    ai-edge-quantizer gives state_in/state_out independent scales, so unless
     *    MODEL_STATE_FEEDBACK_MEMCPY_SAFE, requantise (dequant out -> quant in). */
    for (int k = 0; k < MODEL_N_STATES; k++)
    {
        const model_state_map_t *m = &MODEL_STATE_MAP[k];
        const int8_t *sout = s_interp->output(m->out_pos)->data.int8;
        int8_t *sin = s_interp->input(m->in_pos)->data.int8;
        if (m->memcpy_safe)
        {
            memcpy(sin, sout, (size_t)m->count);
        }
        else
        {
            /* Integer Q16 requant. The naive form -- dequantise to float, then
             * quant_i8() -- costs a SCALAR FLOAT DIVIDE (lrintf(x / scale)) per element,
             * measured at 59 cyc/elem on the HiFi4. With 157,830 state elements and 0/25
             * states memcpy-safe, that was 9.3 M of LiSenNet's 19.2 M cyc/frame -- ~48%
             * of the "inference" cost was this loop, not the network.
             *
             * The scale ratio is loop-invariant, so hoist it once into Q16 fixed point
             * and requantise with a multiply and a shift: 5 cyc/elem. Measured on the
             * ISS: LiSenNet 19,186,677 -> 8,902,092 (2.16x), ConvFSENet 3,640,203 ->
             * 2,767,999 (1.32x), with bit-identical per-frame output checksums.
             *
             * This stays 32-bit ONLY because SE_Init has already proved every ratio is
             * <= SE_STATE_RATIO_MAX (see the note there). On an export whose state input
             * scale collapsed to the calibrator floor (~3.9e-12) the ratio would be ~2.6e14,
             * overflowing int32 and making this loop undefined behaviour -- silently pinning
             * every state to one code and killing the recurrence while the cycle count and
             * the output checksum both look entirely normal. SE_Init refuses such a model
             * rather than letting this loop run on it. */
            const int32_t ratio = (int32_t)(m->out_scale / m->in_scale * 65536.0f + 0.5f);
            for (int i = 0; i < m->count; i++)
            {
                int q = (((((int)sout[i] - m->out_zp) * ratio) + 32768) >> 16) + m->in_zp;
                if (q < -128) q = -128;
                if (q >  127) q =  127;
                sin[i] = (int8_t)q;
            }
        }
    }
    return kStatus_Success;
}

uint32_t SE_ArenaUsedBytes(void) { return s_arenaUsed; }
