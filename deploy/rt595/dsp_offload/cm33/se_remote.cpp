/*
 * se_remote.cpp — M33 side of the LiSenNet SE offload: the RPC core.
 *
 * Exposes the DSP inference as `se_dsp_*` primitives (declared in se_backend.h). Two users:
 *   - production HiFi4-only build: se_remote_alias.c maps SE_* -> se_dsp_* so main.cpp is
 *     unchanged (drop-in for ../../app/model_se_stream.cpp).
 *   - bench build: se_backend_hifi4.cpp wraps se_dsp_* into the se_backend_t vtable so it
 *     can run side-by-side with the M33-local backend.
 * Keeping ONE RPC implementation here avoids drift between the two.
 *
 * Each frame forwards feat -> DSP -> mask over the shared se_ipc mailbox; the int8
 * quant/dequant and the 17-state recurrence live entirely on the DSP.
 */
#include <string.h>

#include "fsl_debug_console.h"
#include "se_backend.h"
#include "se_ipc.h"

/* Same shared block the DSP sees, at the M33 bus alias (placed in dsp_boot.c). */
extern "C" volatile se_ipc_shared_t *g_se_ipc;

/* Cache maintenance for the shared buffers on the M33 side. No-ops if g_se_ipc is in a
 * non-cacheable partition (recommended). Otherwise use the RT500 cache64 ops. */
static inline void ipc_clean_feat(void)      { /* TODO(toolchain-box): XCACHE_CleanByRange(feat)      */ }
static inline void ipc_invalidate_mask(void) { /* TODO(toolchain-box): XCACHE_InvalidateByRange(mask) */ }

static inline void kick_dsp(void) { /* TODO(toolchain-box): MAILBOX kick -> DSP IRQ (bring-up: polling) */ }

static status_t wait_done(void)
{
    for (;;)
    {
        uint32_t s = g_se_ipc->status;
        if (s == SE_IPC_DONE)  return kStatus_Success;
        if (s == SE_IPC_ERROR) return (status_t)g_se_ipc->err_code;
    }
}

extern "C" status_t se_dsp_init(void)
{
    /* DSP_Boot() (dsp_boot.h) must have released the HiFi4. Wait, with a bounded timeout,
     * for the DSP to finish SE_Init() and publish MAGIC. A timeout here is how the bench
     * detects "DSP image not present" and cleanly falls back to M33-only. */
    volatile se_ipc_shared_t *ipc = g_se_ipc;
    uint32_t spins = 0;
    while (ipc->magic != SE_IPC_MAGIC)
    {
        if (ipc->status == SE_IPC_ERROR)
        {
            PRINTF("DSP SE_Init failed (err %u)\r\n", (unsigned)ipc->err_code);
            return (status_t)ipc->err_code;
        }
        if (++spins > 100000000u) return kStatus_Timeout;   /* DSP not up */
    }
    if (ipc->abi_version != SE_IPC_ABI_VERSION || ipc->feature_len != SE_IPC_FEATURE_LEN)
    {
        PRINTF("se_ipc ABI/layout mismatch (abi %u, F %u)\r\n",
               (unsigned)ipc->abi_version, (unsigned)ipc->feature_len);
        return kStatus_Fail;
    }
    se_dsp_reset();
    return kStatus_Success;
}

extern "C" void se_dsp_reset(void)
{
    volatile se_ipc_shared_t *ipc = g_se_ipc;
    ipc->seq   += 1;
    ipc->status = SE_IPC_BUSY;
    ipc->cmd    = SE_IPC_CMD_RESET;
    kick_dsp();
    (void)wait_done();
}

extern "C" status_t se_dsp_process(const float *feat_in, float *mask_out)
{
    volatile se_ipc_shared_t *ipc = g_se_ipc;

    memcpy((void *)ipc->feat, feat_in, sizeof(float) * SE_IPC_FEAT_FLOATS);
    ipc_clean_feat();

    ipc->seq   += 1;
    ipc->status = SE_IPC_BUSY;
    ipc->cmd    = SE_IPC_CMD_PROCESS;
    kick_dsp();

    status_t st = wait_done();
    if (st != kStatus_Success) return st;

    ipc_invalidate_mask();
    memcpy(mask_out, (const void *)ipc->mask, sizeof(float) * SE_IPC_MASK_FLOATS);
    return kStatus_Success;
}

extern "C" uint32_t se_dsp_arena(void)      { return g_se_ipc->arena_used; }
extern "C" uint32_t se_dsp_last_cycles(void){ return g_se_ipc->cycles; }
extern "C" uint32_t se_dsp_clock_hz(void)   { return g_se_ipc->dsp_clock_hz; }
