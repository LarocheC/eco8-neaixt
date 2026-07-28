/*
 * dsp_boot.c — M33: define the shared IPC pointer and release the HiFi4.
 *
 * The two board-specific seams of the whole offload live here and in the linker script:
 *   1. SE_IPC_SHARED_ADDR_M33 — the M33 bus-alias address of the shared SRAM block.
 *   2. DSP_Boot() body        — the SDK call that clocks + un-resets the HiFi4.
 * Both are marked TODO(toolchain-box).
 */
#include "dsp_boot.h"
#include "se_ipc.h"

/* The shared mailbox as the M33 sees it. Must resolve to the SAME physical SRAM the DSP's
 * g_se_ipc points at, expressed in the M33 address view. Set this to a fixed address in a
 * non-cacheable SRAM partition reserved in the M33 linker script (and excluded from the
 * TFLM/heap regions). See README "shared memory". */
#ifndef SE_IPC_SHARED_ADDR_M33
#define SE_IPC_SHARED_ADDR_M33  (0x20200000u)   /* TODO(toolchain-box): real reserved addr */
#endif
volatile se_ipc_shared_t *g_se_ipc = (volatile se_ipc_shared_t *)SE_IPC_SHARED_ADDR_M33;

status_t DSP_Boot(void)
{
    /* Zero the handshake fields so SE_Init()'s wait-on-MAGIC is well-defined even if the
     * block held stale data from a previous run. */
    g_se_ipc->magic  = 0;
    g_se_ipc->status = SE_IPC_BUSY;
    g_se_ipc->cmd    = SE_IPC_CMD_NONE;

    /* TODO(toolchain-box): release the HiFi4.
     * Flow A (MPI): the packed image is already placed; just clock + un-reset the DSP:
     *     BOARD_DSP_Init();                      // from the SDK dsp cm-side helper
     * Flow B (runtime load): copy the embedded dsp_reset/text/data blobs into DSP memory
     *     then release reset. The SDK dsp_examples ".../cm/main.c" shows the exact
     *     SYSCTL0->DSPCFG / clock / reset sequence for evkmimxrt595.
     */
    return kStatus_Success;
}
