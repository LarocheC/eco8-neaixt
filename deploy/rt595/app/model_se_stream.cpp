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
             * 2,767,999 (1.32x), with bit-identical per-frame output checksums. */
            const int ratio = (int)(m->out_scale / m->in_scale * 65536.0f + 0.5f);
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
