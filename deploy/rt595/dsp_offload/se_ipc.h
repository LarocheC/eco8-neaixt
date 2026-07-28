/*
 * se_ipc.h — shared M33 <-> HiFi4 contract for the LiSenNet SE offload.
 *
 * ONE copy of this header is compiled into BOTH cores. It defines the shared-memory
 * mailbox they exchange per frame and the tiny command/status protocol around it.
 *
 * Data path per frame (mirrors SE_ProcessFrame's float in / float out boundary — see
 * ../app/model_se_stream.h):
 *     M33  writes feat[3*F] floats  ->  kicks DSP  ->  DSP runs SE_ProcessFrame
 *     DSP  writes mask[F]  floats   ->  kicks M33  ->  M33 reads mask
 * The int8 quant/dequant and the 17-state recurrence stay entirely inside the DSP, so
 * this interface is the same whether the model is fp32 or int8.
 *
 * PLACEMENT (toolchain box completes): se_ipc_shared_t lives in a fixed physical SRAM
 * block that both cores map. The RT500 M33 and HiFi4 see SRAM at different bus aliases,
 * so the M33 and DSP linker scripts must point their `g_se_ipc` at the SAME physical
 * address (see the SDK rpmsg_lite/multicore shared-mem conventions). Keep the block in a
 * non-cacheable partition, or do explicit cache clean/invalidate at the marked points.
 */
#ifndef SE_IPC_H
#define SE_IPC_H

#include <stdint.h>

/* Kept in sync with ../app/model_io_layout.h (int8 export). If those change, regenerate
 * the headers (host/gen_io_layout.py) and update these two. Static-asserted on both sides. */
#define SE_IPC_FEATURE_LEN   (257)            /* MODEL_FEATURE_LEN  (freq bins)        */
#define SE_IPC_FEAT_FLOATS   (3 * SE_IPC_FEATURE_LEN)   /* 3-channel feature frame     */
#define SE_IPC_MASK_FLOATS   (SE_IPC_FEATURE_LEN)       /* enhanced-magnitude frame    */

#define SE_IPC_MAGIC   (0x53454950u)          /* 'SEIP' — set by DSP once SE_Init done  */

/* status flag, written by DSP, polled/IRQ-woken on M33. */
enum {
    SE_IPC_IDLE  = 0,   /* DSP ready for a command                                      */
    SE_IPC_BUSY  = 1,   /* DSP is processing `cmd`                                       */
    SE_IPC_DONE  = 2,   /* result (mask) valid; M33 may issue the next command          */
    SE_IPC_ERROR = 3,   /* DSP-side failure (see err_code)                              */
};

/* cmd flag, written by M33. */
enum {
    SE_IPC_CMD_NONE    = 0,
    SE_IPC_CMD_PROCESS = 1,   /* run one frame: feat -> mask                            */
    SE_IPC_CMD_RESET   = 2,   /* SE_ResetStates() (between independent utterances)      */
};

/* Cache-line-aligned so feat/mask sit on their own lines for clean/invalidate. */
typedef struct __attribute__((aligned(32))) {
    volatile uint32_t magic;        /* SE_IPC_MAGIC once the DSP model is up            */
    volatile uint32_t abi_version;  /* bump on any layout change; both sides check      */
    volatile uint32_t feature_len;  /* == SE_IPC_FEATURE_LEN; guards header drift       */
    volatile uint32_t arena_used;   /* DSP reports SE_ArenaUsedBytes() after init       */

    volatile uint32_t cmd;          /* M33 -> DSP (SE_IPC_CMD_*)                        */
    volatile uint32_t status;       /* DSP -> M33 (SE_IPC_*)                            */
    volatile uint32_t seq;          /* M33 increments per command; DSP echoes on DONE   */
    volatile uint32_t err_code;     /* DSP status_t on SE_IPC_ERROR                     */

    volatile uint32_t cycles;       /* DSP-side cycle count of the last SE_ProcessFrame */
    volatile uint32_t dsp_clock_hz; /* DSP CCOUNT clock, so the M33 can turn cycles->us  */
    uint32_t _pad[6];               /* pad header to 64 B before the data buffers       */

    float feat[SE_IPC_FEAT_FLOATS]; /* M33 -> DSP: 3-channel feature, channel-major     */
    float mask[SE_IPC_MASK_FLOATS]; /* DSP -> M33: enhanced compressed magnitude        */
} se_ipc_shared_t;

#define SE_IPC_ABI_VERSION  (2u)   /* bumped: added dsp_clock_hz */

#endif /* SE_IPC_H */
