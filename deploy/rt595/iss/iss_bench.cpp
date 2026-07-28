/* Standalone HiFi4 ISS cycle bench for the LiSenNet int8 SE model.
 * Runs the SAME driver (model_se_stream.cpp) + op resolver (model_ops_micro.cpp) the M33
 * firmware uses, but compiled for the RT500 HiFi4 core and executed on the Xtensa ISS.
 * Reports cycles/frame (CCOUNT) over the baked feature frames — the HiFi4 side of the
 * M33-vs-HiFi4 comparison, no board needed. */
#include <stdio.h>
#include <xtensa/tie/xt_timer.h>          /* XT_RSR_CCOUNT */

#include "model_se_stream.h"
#include "model_io_layout.h"
#include "se_test_feats.h"

static float s_mask[MODEL_MASK_LEN];

int main(void)
{
    if (SE_Init() != kStatus_Success) { printf("SE_Init failed\n"); return 1; }
    printf("=== HiFi4 ISS bench: LiSenNet int8 (nxp_rt500 Fusion-F1) ===\n");
    printf("%u IO, %u states, %u-bin features; arena %u B\n",
           (unsigned)MODEL_N_IO, (unsigned)MODEL_N_STATES, (unsigned)MODEL_FEATURE_LEN,
           (unsigned)SE_ArenaUsedBytes());
    printf("frame, cycles, checksum\n");

    SE_ResetStates();
    (void)SE_ProcessFrame(se_test_feats[0], s_mask);   /* warm-up, not timed */
    SE_ResetStates();

    unsigned long total = 0;
    for (unsigned t = 0; t < SE_TEST_FRAMES; t++)
    {
        unsigned c0 = XT_RSR_CCOUNT();
        SE_ProcessFrame(se_test_feats[t], s_mask);
        unsigned cyc = XT_RSR_CCOUNT() - c0;
        total += cyc;
        float s = 0.0f;
        for (int i = 0; i < MODEL_MASK_LEN; i++) s += s_mask[i];
        printf("%u, %u, %d\n", t, cyc, (int)(s * 1000.0f));
    }
    unsigned mean = (unsigned)(total / SE_TEST_FRAMES);
    printf("mean: %u cycles/frame\n", mean);
    /* RT500 M33 budget for real-time at the 16 ms hop is 3.168 M cyc; M33 measured 21.2 M. */
    printf("real-time budget (16ms hop): < ~3.17 M cycles/frame\n");
    return 0;
}
