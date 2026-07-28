/*
 * dsp_main.cpp — HiFi4 (Fusion F1) side of the LiSenNet SE offload.
 *
 * Runs the SAME TFLite-Micro streaming driver as the M33 bring-up firmware
 * (../../app/model_se_stream.cpp + model_ops_micro.cpp + the int8 model_data.h /
 * model_io_layout.h) — but this translation unit is built with the Xtensa toolchain and
 * linked against middleware/eiq/tensorflow-lite/lib/hifi4/xcc/rt685s/libtflm.a, so every
 * Conv2D / DepthwiseConv2D / TransposeConv / Logistic / elementwise op dispatches to the
 * xa_nnlib HiFi4 int8 kernels instead of CMSIS-NN. Nothing model-specific changes here;
 * the speedup is entirely the linked kernel library.
 *
 * Role: secondary core. The M33 boots us (MPI loader), then we own the model + arena +
 * the 17-state recurrence and service one frame per SE_IPC_CMD_PROCESS.
 */
#include <string.h>

#include "fsl_common.h"
#include "model_se_stream.h"     /* SE_Init / SE_ResetStates / SE_ProcessFrame / SE_ArenaUsedBytes */
#include "model_io_layout.h"     /* MODEL_FEATURE_LEN, MODEL_MASK_LEN                              */
#include "se_ipc.h"

/* Compile-time guard that the shared contract still matches the generated model layout. */
static_assert(SE_IPC_FEATURE_LEN == MODEL_FEATURE_LEN, "se_ipc feature_len drift vs model_io_layout.h");
static_assert(SE_IPC_MASK_FLOATS == MODEL_MASK_LEN,    "se_ipc mask_len drift vs model_io_layout.h");

/* The shared mailbox. Both cores resolve `g_se_ipc` to the same physical SRAM block via
 * their linker scripts (see se_ipc.h + README_DSP_OFFLOAD.md "shared memory"). */
extern "C" volatile se_ipc_shared_t *g_se_ipc;

/* --- HiFi4 cycle counter (Xtensa CCOUNT) ------------------------------------------- */
static inline uint32_t dsp_ccount(void)
{
    uint32_t r;
    __asm__ __volatile__("rsr.ccount %0" : "=a"(r));   /* CCOUNT increments at DSP clock */
    return r;
}

/* --- cache maintenance for the shared buffers -------------------------------------- *
 * No-ops if g_se_ipc lives in a non-cacheable SRAM partition (recommended). If it is
 * cacheable, replace with the Xtensa dcache ops (xthal_dcache_region_invalidate /
 * _writeback) — see README "cache coherency". */
static inline void ipc_invalidate_feat(void) { /* TODO(toolchain-box) if cacheable */ }
static inline void ipc_writeback_mask(void)   { /* TODO(toolchain-box) if cacheable */ }

/* --- inter-core kick ---------------------------------------------------------------- *
 * Wait for the M33 to signal a new command, and kick the M33 when a result is ready.
 * Bring-up default is polling (works with zero IRQ wiring). For low-latency streaming,
 * wire the RT500 MAILBOX peripheral: M33 write -> DSP IRQ, DSP write -> M33 IRQ. */
static inline void kick_m33(void)   { /* TODO(toolchain-box): MAILBOX_SetValue(...) to raise the M33 IRQ */ }

int main(void)
{
    /* Board/DSP clock + cache init comes from the DSP board overlay (BOARD_InitHardware
     * in the dsp core template). Minimal here; the MPI loader has already placed us. */
    extern void BOARD_InitHardware(void);
    BOARD_InitHardware();

    volatile se_ipc_shared_t *ipc = g_se_ipc;
    ipc->magic  = 0;
    ipc->status = SE_IPC_BUSY;          /* not ready until the model is allocated */

    if (SE_Init() != kStatus_Success)
    {
        ipc->err_code = (uint32_t)kStatus_Fail;
        ipc->status   = SE_IPC_ERROR;
        for (;;) {}
    }
    ipc->feature_len = MODEL_FEATURE_LEN;
    ipc->arena_used  = SE_ArenaUsedBytes();
    /* Publish the DSP CCOUNT clock so the M33 can convert ipc->cycles to microseconds.
     * TODO(toolchain-box): query the real DSP clock (SDK CLOCK_GetDspClkFreq() or equiv);
     * default assumes the DSP runs at the M33's 198 MHz. */
#ifndef DSP_CLOCK_HZ
#define DSP_CLOCK_HZ (198000000u)
#endif
    ipc->dsp_clock_hz = DSP_CLOCK_HZ;
    ipc->abi_version = SE_IPC_ABI_VERSION;
    ipc->err_code    = 0;
    ipc->cmd         = SE_IPC_CMD_NONE;
    ipc->status      = SE_IPC_IDLE;
    __asm__ __volatile__("memw");       /* publish header before MAGIC */
    ipc->magic       = SE_IPC_MAGIC;    /* M33 spins on this to know the DSP is up */

    /* Service loop: one SE_ProcessFrame per PROCESS command. */
    for (;;)
    {
        uint32_t cmd = ipc->cmd;
        if (cmd == SE_IPC_CMD_NONE)
            continue;                   /* replace the spin with WAITI + MAILBOX IRQ later */

        ipc->cmd    = SE_IPC_CMD_NONE;
        ipc->status = SE_IPC_BUSY;

        if (cmd == SE_IPC_CMD_RESET)
        {
            SE_ResetStates();
            ipc->status = SE_IPC_DONE;
            kick_m33();
            continue;
        }

        /* SE_IPC_CMD_PROCESS */
        ipc_invalidate_feat();
        static float feat[SE_IPC_FEAT_FLOATS];
        static float mask[SE_IPC_MASK_FLOATS];
        memcpy(feat, (const void *)ipc->feat, sizeof(feat));

        uint32_t t0 = dsp_ccount();
        status_t st = SE_ProcessFrame(feat, mask);   /* the accelerated inference */
        ipc->cycles = dsp_ccount() - t0;

        if (st != kStatus_Success)
        {
            ipc->err_code = (uint32_t)st;
            ipc->status   = SE_IPC_ERROR;
            kick_m33();
            continue;
        }

        memcpy((void *)ipc->mask, mask, sizeof(mask));
        ipc_writeback_mask();
        ipc->seq   += 1;
        ipc->status = SE_IPC_DONE;
        kick_m33();
    }
}
