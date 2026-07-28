/*
 * dsp_boot.h — M33 brings the HiFi4 out of reset before SE_Init().
 *
 * main.cpp gains exactly one line: `DSP_Boot();` right after BOARD_Init().
 * Everything else (SE_Init/SE_ResetStates/SE_ProcessFrame) is unchanged — se_remote.cpp
 * turns those into DSP RPCs.
 */
#ifndef DSP_BOOT_H
#define DSP_BOOT_H

#include "fsl_common.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* Load the HiFi4 image and release its reset.
 *
 * TWO supported flows (pick one in dsp_boot.c — see README_DSP_OFFLOAD.md):
 *   (A) MPI / mpi_loader: the DSP image is packed with the M33 image (elf2sb) and the
 *       secondary bootloader places it; DSP_Boot() only asserts the DSP clock + releases
 *       reset (SYSCTL). This is the flash-deploy path.
 *   (B) Runtime load: the DSP reset/text/data blobs are embedded in the M33 image and
 *       copied into DSP TCM here, then reset released. Handy for SWD bring-up.
 *
 * Returns kStatus_Success once the DSP core is running (SE_Init() then waits on MAGIC). */
status_t DSP_Boot(void);

#if defined(__cplusplus)
}
#endif

#endif /* DSP_BOOT_H */
