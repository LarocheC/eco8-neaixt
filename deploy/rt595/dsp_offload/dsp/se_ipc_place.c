/*
 * se_ipc_place.c — DSP: define the shared IPC pointer at the HiFi4 bus alias.
 *
 * Same physical SRAM as the M33's g_se_ipc, but expressed in the DSP address view (the
 * RT500 HiFi4 sees SRAM at different addresses than the M33). Set from the DSP linker
 * script's reserved shared region. See README "shared memory".
 */
#include "se_ipc.h"

#ifndef SE_IPC_SHARED_ADDR_DSP
#define SE_IPC_SHARED_ADDR_DSP  (0x00200000u)   /* TODO(toolchain-box): DSP-alias of the same block */
#endif
volatile se_ipc_shared_t *g_se_ipc = (volatile se_ipc_shared_t *)SE_IPC_SHARED_ADDR_DSP;
