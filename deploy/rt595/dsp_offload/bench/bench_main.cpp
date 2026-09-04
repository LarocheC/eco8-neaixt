/*
 * bench_main.cpp — A/B benchmark: LiSenNet SE frames through the M33-local backend and the
 * HiFi4-offloaded backend, on the same board in one run, over the same baked feature frames
 * (../../app/se_test_feats.h). Reports, per backend:
 *   - M33-observed latency per frame (DWT cycles + us) — for HiFi4 this is the full RPC
 *     round-trip (marshal feat, kick, DSP compute, kick back, read mask): the real cost.
 *   - HiFi4 only: the DSP's own compute cycles/us (CCOUNT), so raw kernel speed is separated
 *     from RPC overhead.
 *   - a per-frame output checksum, cross-checked M33 vs HiFi4 to confirm the offload is
 *     numerically faithful (small int8 CMSIS-NN vs xa_nnlib rounding differences aside).
 *
 * Degrades gracefully: if the DSP image isn't present (se_dsp_init times out), the HiFi4
 * backend is skipped and you still get the M33 int8 baseline. So this is useful BEFORE the
 * DSP half is built — it gives the int8 M33 numbers (vs the current fp32 firmware) on its own.
 *
 * Replaces ../../app/main.cpp in the bench build (see CMakeLists.snippet.txt).
 */
#include <stdint.h>

#include "board.h"
#include "fsl_clock.h"
#include "fsl_debug_console.h"

extern "C" void BOARD_Init(void);
#include "dsp_boot.h"
#include "se_backend.h"
#include "model_io_layout.h"     /* MODEL_MASK_LEN, MODEL_FEATURE_LEN */
#include "se_test_feats.h"       /* se_test_feats[SE_TEST_FRAMES][3*257], SE_TEST_FRAMES */

/* DWT cycle counter (Cortex-M33). Measures the M33-observed latency of each process() call. */
static void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}
static inline uint32_t dwt_cycles(void) { return DWT->CYCCNT; }

static inline uint32_t cyc_to_us(uint32_t cyc, uint32_t hz)
{
    return hz ? (uint32_t)((uint64_t)cyc * 1000000u / hz) : 0u;
}

/* Run one backend over the whole sequence; fills per-frame checksums and returns mean
 * M33-latency cycles (0 if the backend was unavailable). */
static uint32_t run_backend(const se_backend_t *b, uint32_t core_hz,
                            int32_t *checksum_out /* [SE_TEST_FRAMES] */,
                            int *available_out)
{
    *available_out = 0;
    if (b->init() != kStatus_Success)
    {
        PRINTF("\r\n[%s] unavailable — skipped\r\n", b->name);
        return 0;
    }
    *available_out = 1;
    const uint32_t dsp_hz = b->compute_clock_hz();
    PRINTF("\r\n=== %s ===\r\n", b->name);
    PRINTF("arena %u B\r\n", (unsigned)b->arena_used());
    PRINTF("frame, m33_cyc, m33_us%s, checksum\r\n",
           dsp_hz ? ", dsp_cyc, dsp_us" : "");

    static float mask[MODEL_MASK_LEN];
    b->reset();
    b->process(se_test_feats[0], mask);   /* warm-up frame, not timed */
    b->reset();

    uint64_t sum_m33 = 0, sum_dsp = 0;
    for (uint32_t t = 0; t < SE_TEST_FRAMES; t++)
    {
        uint32_t c0 = dwt_cycles();
        b->process(se_test_feats[t], mask);
        uint32_t m33_cyc = dwt_cycles() - c0;          /* M33-observed latency */
        uint32_t dsp_cyc = b->last_compute_cycles();   /* DSP compute (0 for M33 backend) */

        float s = 0.0f;
        for (int i = 0; i < MODEL_MASK_LEN; i++) s += mask[i];
        checksum_out[t] = (int32_t)(s * 1000.0f);

        sum_m33 += m33_cyc;
        sum_dsp += dsp_cyc;
        if (dsp_hz)
            PRINTF("%u, %u, %u, %u, %u, %d\r\n", (unsigned)t, (unsigned)m33_cyc,
                   (unsigned)cyc_to_us(m33_cyc, core_hz), (unsigned)dsp_cyc,
                   (unsigned)cyc_to_us(dsp_cyc, dsp_hz), (int)checksum_out[t]);
        else
            PRINTF("%u, %u, %u, %d\r\n", (unsigned)t, (unsigned)m33_cyc,
                   (unsigned)cyc_to_us(m33_cyc, core_hz), (int)checksum_out[t]);
    }

    uint32_t mean_m33 = (uint32_t)(sum_m33 / SE_TEST_FRAMES);
    PRINTF("mean: %u M33-cyc/frame (%u us observed latency)\r\n",
           (unsigned)mean_m33, (unsigned)cyc_to_us(mean_m33, core_hz));
    if (dsp_hz)
    {
        uint32_t mean_dsp = (uint32_t)(sum_dsp / SE_TEST_FRAMES);
        PRINTF("      %u DSP-cyc/frame compute (%u us); RPC overhead ~%u us/frame\r\n",
               (unsigned)mean_dsp, (unsigned)cyc_to_us(mean_dsp, dsp_hz),
               (unsigned)(cyc_to_us(mean_m33, core_hz) - cyc_to_us(mean_dsp, dsp_hz)));
    }
    return mean_m33;
}

int main(void)
{
    BOARD_Init();
    DSP_Boot();          /* release the HiFi4; harmless/no-op if this is an M33-only image */
    dwt_init();

    const uint32_t core_hz = CLOCK_GetFreq(kCLOCK_CoreSysClk);
    PRINTF("\r\n=== RT595 LiSenNet SE bench: M33 vs HiFi4 (int8) ===\r\n");
    PRINTF("core %u MHz; %u frames; %u-bin features; real-time budget < %u cyc/frame (16 ms hop)\r\n",
           (unsigned)(core_hz / 1000000u), (unsigned)SE_TEST_FRAMES, (unsigned)MODEL_FEATURE_LEN,
           (unsigned)((uint64_t)core_hz * 16u / 1000u));

    static int32_t chk_m33[SE_TEST_FRAMES];
    static int32_t chk_dsp[SE_TEST_FRAMES];
    int have_m33 = 0, have_dsp = 0;

    uint32_t m33_mean = run_backend(&se_backend_m33,   core_hz, chk_m33, &have_m33);
    uint32_t dsp_mean = run_backend(&se_backend_hifi4, core_hz, chk_dsp, &have_dsp);

    PRINTF("\r\n=== summary ===\r\n");
    if (have_m33 && have_dsp && dsp_mean)
    {
        /* effective speedup = M33 compute latency / HiFi4 round-trip latency (both M33-cyc). */
        PRINTF("effective speedup (incl. RPC): %u.%02ux\r\n",
               (unsigned)(m33_mean / dsp_mean),
               (unsigned)(((uint64_t)m33_mean * 100u / dsp_mean) % 100u));

        int match = 1, maxdiff = 0;
        for (uint32_t t = 0; t < SE_TEST_FRAMES; t++)
        {
            int d = chk_m33[t] - chk_dsp[t]; if (d < 0) d = -d;
            if (d > maxdiff) maxdiff = d;
            if (chk_m33[t] != chk_dsp[t]) match = 0;
        }
        PRINTF("checksum cross-check: %s (max |diff| = %d over %u frames)\r\n",
               match ? "IDENTICAL" : "close", maxdiff, (unsigned)SE_TEST_FRAMES);
    }
    else if (have_m33)
    {
        PRINTF("HiFi4 not present — M33 int8 baseline only. Build+flash the DSP image (MPI)\r\n"
               "then re-run for the A/B. See README_DSP_OFFLOAD.md.\r\n");
    }
    PRINTF("done.\r\n");
    for (;;) {}
}
