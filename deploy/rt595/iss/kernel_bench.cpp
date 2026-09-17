/* Per-op HiFi4 cost model: cycles for ONE int8 kernel at a realistic streaming shape.
 *
 * Every HiFi4 figure in BENCHMARKS.md is a whole-model number, and every model measured
 * so far is a dense MLP. That leaves the question the architecture work actually needs
 * answered -- what does xa_nnlib's int8 CONV_2D cost per MAC on this core? -- unmeasured.
 * This runs each single-op model from host/gen_kernel_bench.py through the same TFLM
 * interpreter the firmware uses and reports cycles, arena, and MAC/cycle.
 *
 * Timing note: built with -mlsp=sim, so memory is zero-wait-state. Per BENCHMARKS.md the
 * silicon capture showed that IS the hardware number for an image that fits DSP local SRAM
 * (1.001x on blockdiag_full). These are single ops with small tensors, so they fit; the
 * whole-model fit check stays a separate gate.
 */
#include <stdio.h>
#include <string.h>
#include <xtensa/tie/xt_timer.h>

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "kernel_bench_models.h"

#define ARENA_BYTES (600 * 1024)
static uint8_t g_arena[ARENA_BYTES] __attribute__((aligned(16)));

#define REPS 5

static tflite::MicroOpResolver &Resolver(void)
{
    static tflite::MicroMutableOpResolver<13> r;
    static bool init = false;
    if (!init) {
        r.AddConv2D(); r.AddDepthwiseConv2D(); r.AddFullyConnected();
        r.AddBatchMatMul(); r.AddQuantize(); r.AddDequantize();
        r.AddReshape(); r.AddTranspose();
        /* No AddMul(): its float path pulls xa_nn_vec_activation_min_max_f32_f32
         * out of a section --orphan-handling=discard drops, and nothing here
         * needs MUL. ReLU is a fused conv activation, so no AddRelu() either. */
        r.AddAdd(); r.AddLogistic(); r.AddMean();
        init = true;
    }
    return r;
}

static int RunOne(const KernelBenchModel &m)
{
    const tflite::Model *model = tflite::GetModel(m.data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("%s, SCHEMA_MISMATCH\n", m.name);
        return 1;
    }
    tflite::MicroInterpreter interp(model, Resolver(), g_arena, ARENA_BYTES);
    if (interp.AllocateTensors() != kTfLiteOk) {
        printf("%s, ALLOC_FAILED (arena %d B too small?)\n", m.name, ARENA_BYTES);
        return 1;
    }

    /* Deterministic non-trivial input: zeros can let a kernel short-circuit.
     * Fill EVERY input -- a streaming candidate carries one FIFO window per block,
     * and an unwritten state tensor would time a different computation. */
    for (size_t t = 0; t < interp.inputs_size(); t++) {
        TfLiteTensor *in = interp.input(t);
        for (size_t i = 0; i < in->bytes; i++)
            in->data.int8[i] = (int8_t)((i * 37 + 11 + t * 7) & 0x7f) - 64;
    }

    if (interp.Invoke() != kTfLiteOk) { printf("%s, INVOKE_FAILED\n", m.name); return 1; }

    unsigned best = 0xffffffffu, total = 0;
    for (int r = 0; r < REPS; r++) {
        unsigned c0 = XT_RSR_CCOUNT();
        interp.Invoke();
        unsigned c = XT_RSR_CCOUNT() - c0;
        if (c < best) best = c;
        total += c;
    }

    /* Checksum so a shape that silently computes nothing is visible. */
    const TfLiteTensor *out = interp.output(0);
    long sum = 0;
    for (size_t i = 0; i < out->bytes; i++) sum += out->data.int8[i];

    unsigned arena = (unsigned)interp.arena_used_bytes();
    const int nops = (int)model->subgraphs()->Get(0)->operators()->size();
    /* MAC/cycle x1000, printed as an integer: no float formatting in the sim console. */
    unsigned mac_per_kcyc = (unsigned)((m.macs * 1000LL) / (long long)(best ? best : 1));
    printf("%s, %lld, %u, %u, %u, %d, %u, %ld\n",
           m.name, m.macs, best, total / REPS, arena, nops, mac_per_kcyc, sum);
    return 0;
}

int main(void)
{
    printf("=== HiFi4 per-op int8 kernel cost (nxp_rt500 Fusion-F1, TFLM + xa_nnlib) ===\n");
    printf("arena cap %d B; %d timed reps per op; budget 3,168,000 cyc/frame @198MHz\n",
           ARENA_BYTES, REPS);
    printf("name, macs, cyc_min, cyc_mean, arena_b, n_ops, mac_per_1000cyc, out_checksum\n");
    int bad = 0;
    for (int i = 0; i < kKernelBenchModelCount; i++) bad += RunOne(kKernelBenchModels[i]);
    printf("done: %d models, %d failed\n", kKernelBenchModelCount, bad);
    return 0;
}
