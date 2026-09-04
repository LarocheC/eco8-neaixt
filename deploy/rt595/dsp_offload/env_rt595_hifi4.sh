#!/bin/bash
# env_rt595_hifi4.sh — Cadence Xtensa toolchain environment for building the RT595 HiFi4
# DSP target (deploy/rt595/dsp_offload/dsp/).
#
# Adapted from a working Cadence RI-2023.11 setup for a HiFi5 target. The only
# substantive change for RT595 is the core: a HiFi5 core config -> the NXP RT500
# HiFi4 core. Same license model (FlexLM daemon), same tools layout.
#
# Source this on a machine that HAS the Cadence tools + a running license daemon, THEN run
# the west/mcux build from README_DSP_OFFLOAD.md. (The training box has neither — it is not
# an Xtensa build host.)

# --- FlexLM license --------------------------------------------------------------------
# Set this to your own Cadence license source before sourcing: either "port@host" for a
# license server, or an absolute path to a local .lic file. There is no default; the
# build host must supply it.
: "${XTENSA_LICENSE_FILE:?set XTENSA_LICENSE_FILE to port@host or a path to your Cadence .lic}"
export XTENSA_LICENSE_FILE

# --- Xtensa tools (unchanged layout; RI-2023.11 as the overlay uses) ---------------------
export XTENSA_ROOT=${HOME}/toolchains
export XTENSA_VER=RI-2023.11-linux
export XTENSA_SYSTEM=$XTENSA_ROOT/tools/$XTENSA_VER/XtensaTools/config/
export XTENSA_BIN_PATH=$XTENSA_ROOT/tools/$XTENSA_VER/XtensaTools/bin
export PATH="${PATH}:$XTENSA_BIN_PATH"

# --- Core: the one real change vs a HiFi5 environment ------------------------------------
# RT595 == RT500 family; the SDK's DSP examples export this HiFi4 core name.
export XTENSA_CORE="nxp_rt600_RI2021_8_newlib"
export XTENSA_TESTBENCH_INC_PATH=$XTENSA_ROOT/builds/$XTENSA_VER/$XTENSA_CORE

# --- What the NXP west/mcux build reads (sdk/cmake/toolchain/xtensa.cmake) ---------------
export XCC_DIR=$XTENSA_BIN_PATH                    # xtensa.cmake wants XCC_DIR

# ---------------------------------------------------------------------------------------
# PREREQUISITES this script does NOT and cannot create (all Cadence/NXP license-gated):
#   1. The RI-2023.11 XtensaTools installed at $XTENSA_ROOT/tools/$XTENSA_VER  (Cadence).
#   2. A FlexLM license daemon running at $XTENSA_LICENSE_FILE with a Cadence license.
#   3. The RT500 HiFi4 core "$XTENSA_CORE" REGISTERED in this XtensaTools install and
#      the license ENTITLED to it. A HiFi5 build host will have a HiFi5 core, not this
#      one — obtain it from NXP (MCUXpresso "Xtensa configuration for RT500/RT600 DSP")
#      and `xt-regfile -r` it; confirm the license feature covers it.
#
# VERSION NOTE: the core name encodes RI-2021.8, but the tools are RI-2023.11. Newer tools
# usually consume older core configs; verify with `xt-clang --show-config=core` after
# sourcing. If it rejects the core, either install matching RI-2021.8 tools or get the
# RT500 core rebuilt for RI-2023.11 from NXP.
#
# Quick checks after sourcing:
#   xt-run --show-config=configs      # is $XTENSA_CORE registered?
#   lmutil lmstat -a -c "$XTENSA_LICENSE_FILE"   # is the license reachable + entitled?
#   xt-clang --version
echo "RT595 HiFi4 Xtensa env: core=$XTENSA_CORE tools=$XTENSA_VER lic=$XTENSA_LICENSE_FILE"
