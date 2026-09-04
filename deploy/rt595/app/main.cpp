/*
 * RT595 M33 bring-up app for the LiSenNet streaming SE model on TFLite-Micro.
 *
 * Feeds a fixed sequence of feature frames (se_test_feats.h) through the streaming
 * driver and reports per-frame latency + an output checksum over the debug UART
 * (ST-LINK/LPC VCP -> /dev/ttyACM0). This mirrors the N6 firmware_router bring-up
 * (baked feats, UART telemetry); live audio + codec I/O is a later step.
 *
 * Board init (BOARD_Init*) and the cycle timer come from the mcuxsdk board overlay
 * for evkmimxrt595@cm33 (added when this is wired into the SDK example tree).
 */
#include <stdint.h>

#include "board.h"
#include "fsl_clock.h"
#include "fsl_debug_console.h"

/* BOARD_Init() is defined in the board overlay's hardware_init.c (C linkage). */
extern "C" void BOARD_Init(void);

#include "model_se_stream.h"
#include "model_io_layout.h"
#include "se_test_feats.h"

/* DWT cycle counter for per-frame latency (Cortex-M33 CoreDebug). */
static void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}
static inline uint32_t dwt_cycles(void) { return DWT->CYCCNT; }

int main(void)
{
    BOARD_Init();     /* pins + clocks + debug console (board overlay) */
    dwt_init();

    PRINTF("\r\n=== RT595 %s streaming SE (M33 / TFLite-Micro) ===\r\n", MODEL_NAME);

    if (SE_Init() != kStatus_Success)
    {
        PRINTF("SE_Init failed\r\n");
        for (;;) {}
    }

    const uint32_t core_hz = CLOCK_GetFreq(kCLOCK_CoreSysClk);
    PRINTF("core %u MHz; %u test frames of %u-bin features\r\n",
           (unsigned)(core_hz / 1000000u), (unsigned)SE_TEST_FRAMES, (unsigned)MODEL_FEATURE_LEN);
    PRINTF("frame, cycles, us, mask_checksum\r\n");

    static float mask_out[MODEL_MASK_LEN];
    SE_ResetStates();

    for (uint32_t t = 0; t < SE_TEST_FRAMES; t++)
    {
        uint32_t c0 = dwt_cycles();
        if (SE_ProcessFrame(se_test_feats[t], mask_out) != kStatus_Success)
        {
            PRINTF("frame %u failed\r\n", (unsigned)t);
            break;
        }
        uint32_t cyc = dwt_cycles() - c0;

        /* Cheap output signature so the host can confirm determinism run-to-run. */
        float sum = 0.0f;
        for (int i = 0; i < MODEL_MASK_LEN; i++) sum += mask_out[i];
        uint32_t us = (uint32_t)((uint64_t)cyc * 1000000u / core_hz);
        PRINTF("%u, %u, %u, %d\r\n", (unsigned)t, (unsigned)cyc, (unsigned)us, (int)(sum * 1000.0f));
    }

    PRINTF("done. RTF target: cycles/frame < %u for real-time at 16 ms hop\r\n",
           (unsigned)((uint64_t)core_hz * 16u / 1000u));
    for (;;) {}
}
