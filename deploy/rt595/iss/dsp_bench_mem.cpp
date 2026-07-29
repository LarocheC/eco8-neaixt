/* HiFi4 on-silicon cycle bench — reports through shared memory, not stdio.
 *
 * The ISS bench (iss_bench.cpp) printf()s its results, which works under xt-run and
 * under the gdbio LSP. Neither is available on real hardware: the deployable LSP is
 * min-rt, whose libgloss stubs SILENTLY DROP stdio. So on silicon the bench must hand
 * its numbers back some other way.
 *
 * It writes them into a fixed struct in DSP data RAM. Two readers can pick them up:
 *   * the M33, via the SRAM alias, or
 *   * a debugger over SWD (what we actually do -- pyocd reads the struct directly,
 *     so no IPC, no mailbox and no M33-side polling code is needed).
 *
 * `magic` is written LAST and acts as the done-flag: a reader that sees SE_BENCH_MAGIC
 * knows every other field is complete. `state` advances so a hung run can be told apart
 * from a crashed one.
 */
#include <stdint.h>
#include <xtensa/tie/xt_timer.h>          /* XT_RSR_CCOUNT */

#include "model_se_stream.h"
#include "model_io_layout.h"
#include "se_test_feats.h"

#define SE_BENCH_MAGIC 0x48465F31u        /* "HF_1" */

/* Progress markers, so a reader can locate a failure without any console. */
enum {
    ST_ENTER      = 1,
    ST_INIT_OK    = 2,
    ST_INIT_FAIL  = 0xE1,
    ST_WARMED     = 3,
    ST_RUNNING    = 4,
    ST_DONE       = 5,
    ST_FRAME_FAIL = 0xE2,
};

typedef struct {
    volatile uint32_t magic;              /* written LAST; SE_BENCH_MAGIC == complete */
    volatile uint32_t state;              /* ST_* above                               */
    volatile uint32_t n_frames;
    volatile uint32_t arena_used;
    volatile uint32_t n_io;
    volatile uint32_t n_states;
    volatile uint32_t feat_len;
    volatile uint32_t ccount_overhead;    /* cost of reading CCOUNT itself            */
    volatile uint32_t cycles[SE_TEST_FRAMES];
    volatile int32_t  checksum[SE_TEST_FRAMES];
} se_bench_result_t;

/* An ordinary global, deliberately NOT in a custom section: the link uses
 * --orphan-handling=discard (needed to absorb a bogus /DISCARD/ in the TFLM archive),
 * which would throw a custom section away. It lands in .bss, so the host finds its
 * address with `xt-nm dsp_bench.elf | grep g_bench_result` and reads it over SWD.
 * Being zero-initialised is exactly right: `magic` only becomes valid once written. */
__attribute__((used))
volatile se_bench_result_t g_bench_result;

static float s_mask[MODEL_MASK_LEN];

int main(void)
{
    g_bench_result.magic = 0u;                 /* invalidate before touching anything */
    g_bench_result.state = ST_ENTER;

    if (SE_Init() != kStatus_Success)
    {
        g_bench_result.state = ST_INIT_FAIL;
        g_bench_result.magic = SE_BENCH_MAGIC;  /* publish the failure too */
        for (;;) {}
    }

    g_bench_result.state      = ST_INIT_OK;
    g_bench_result.arena_used = SE_ArenaUsedBytes();
    g_bench_result.n_io       = MODEL_N_IO;
    g_bench_result.n_states   = MODEL_N_STATES;
    g_bench_result.feat_len   = MODEL_FEATURE_LEN;
    g_bench_result.n_frames   = SE_TEST_FRAMES;

    /* Measure the counter-read overhead so it can be subtracted if it ever matters. */
    {
        unsigned a = XT_RSR_CCOUNT();
        unsigned b = XT_RSR_CCOUNT();
        g_bench_result.ccount_overhead = b - a;
    }

    SE_ResetStates();
    (void)SE_ProcessFrame(se_test_feats[0], s_mask);   /* warm-up, untimed */
    SE_ResetStates();
    g_bench_result.state = ST_WARMED;

    g_bench_result.state = ST_RUNNING;
    for (unsigned t = 0; t < SE_TEST_FRAMES; t++)
    {
        unsigned c0 = XT_RSR_CCOUNT();
        status_t st = SE_ProcessFrame(se_test_feats[t], s_mask);
        unsigned cyc = XT_RSR_CCOUNT() - c0;
        if (st != kStatus_Success)
        {
            g_bench_result.state = ST_FRAME_FAIL;
            g_bench_result.magic = SE_BENCH_MAGIC;
            for (;;) {}
        }
        /* Same checksum expression as iss_bench.cpp:37 and app/main.cpp:69, so silicon,
         * ISS and M33 numbers are all directly comparable. */
        float s = 0.0f;
        for (int i = 0; i < MODEL_MASK_LEN; i++) s += s_mask[i];
        g_bench_result.cycles[t]   = cyc;
        g_bench_result.checksum[t] = (int32_t)(s * 1000.0f);
    }

    g_bench_result.state = ST_DONE;
    g_bench_result.magic = SE_BENCH_MAGIC;     /* publish: everything above is valid */

    for (;;) {}
}
