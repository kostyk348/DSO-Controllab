#ifndef DSO_CONTROLLERS_H
#define DSO_CONTROLLERS_H

#include <stdint.h>
#include "plant.h"
#include "gp.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Controller API
 *
 * All controllers run simulation on a plant, return result.
 * Resource metrics stored by pointer.
 * ================================================================ */

typedef struct {
    double iae;            /* integral absolute error */
    double itae;           /* time-weighted IAE: sum(t * |error| * dt) */
    double overshoot;
    double energy;         /* mean(u^2) over steps */
    double settling_time;  /* seconds until error stays within ±2% */
    int    saturated;      /* count of steps where |u| >= 3.999 */
    double score;          /* combined: itae/T + 0.3*os + 0.08*energy + 0.005*settle + 0.12*wcet + 0.9*jitter */
    double wcet_us;
    double jitter_us;
    int    cycles;
    int    ram_bytes;
    int    branch_points;
    int    controller_tier; /* 0=GP, 1=LQR, 2=PID, 3=LeadLag, -1=unknown */
} ControllerResult;

/* ─── Resource metrics ────────────────────────────────────────── */
static inline void controller_resource_metrics(int cycles, int ram_bytes, int branch_points,
                                                double *out_wcet, double *out_jitter) {
    (void)ram_bytes;
    *out_wcet   = cycles / 48.0;
    *out_jitter = 0.04 + 0.055 * branch_points;
}

/* ─── Score (itae-centered) ───────────────────────────────────── */
static inline double controller_score(double itae, double total_time,
                                       double overshoot, double energy,
                                       double settling_time,
                                       double wcet_us, double jitter_us) {
    double itae_norm = itae / total_time;
    double settle_penalty = 0.005 * settling_time;
    return itae_norm + 0.30 * overshoot + 0.08 * energy
         + settle_penalty
         + 0.12 * wcet_us + 0.9 * jitter_us;
}

/* ─── Clamp ───────────────────────────────────────────────────── */
static inline double clamp_val(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

/* ================================================================
 * PID controller
 * ================================================================ */
ControllerResult pid_simulate(double kp, double ki, double kd,
                               const Plant *plant, int steps, double dt,
                               uint64_t *seed);

/* ================================================================
 * LQR controller
 * ================================================================ */
void lqr_compute_gain(const Plant *plant, double dt, double K[2]);
ControllerResult lqr_simulate(const double K[2],
                               const Plant *plant, int steps, double dt,
                               uint64_t *seed);

/* ================================================================
 * MPC controller
 * ================================================================ */
ControllerResult mpc_simulate(const Plant *plant, int steps, double dt,
                               uint64_t *seed);

/* ================================================================
 * Lead-Lag compensator (ultra-light: 2 cycles, 0 branches)
 *
 *   C(s) = k * (s + z) / (s + p)
 *   Discretized via Tustin, implemented as difference equation:
 *     u[k] = b0*e[k] + b1*e[k-1] - a1*u[k-1]
 * ================================================================ */
ControllerResult leadlag_simulate(double k, double z, double p,
                                   const Plant *plant, int steps, double dt,
                                   uint64_t *seed);

/* ================================================================
 * DSO — DataSpace OS Controller Runtime
 *
 * Architecture:
 *   Tier 1: GP library (top-K evolved trees, lowest resource)
 *   Tier 2: LQR (computed via DARE, moderate resource)
 *   Tier 3: PID (best-of-36 search, simple & reliable)
 *   Tier 4: Lead-Lag (2 cycles, 0 branches, failsafe)
 *
 * Plant fingerprinting:
 *   Quantize (wn, zeta, gain, delay) → 64-bit hash
 *   Cache best controller tier per fingerprint
 *   Next encounter → skip directly to known best
 *
 * Online learning:
 *   After each deployment, update bank with actual score
 *   If performance degrades, fall back to next tier
 * ================================================================ */

/* Plant fingerprint: hash of quantized plant parameters */
typedef uint64_t DsoFingerprint;
DsoFingerprint dso_fingerprint(const Plant *p);

#define DSO_BANK_MAX 128

/* Bank entry: learned association fingerprint → best controller */
typedef struct {
    DsoFingerprint fp;
    int  tier;              /* 0=GP, 1=LQR, 2=PID, 3=LeadLag */
    int  gp_idx;            /* which GP tree in library (-1 if none) */
    double K_lqr[2];        /* LQR gain */
    double kp, ki, kd;      /* PID gains */
    double k_lead, z_lead, p_lead; /* Lead-Lag params */
    double best_score;      /* best score achieved */
    int    hit_count;       /* times deployed */
} DsoBankEntry;

/* DSO runtime configuration */
typedef struct {
    /* GP library (evolved trees, sorted best-first) */
    GpTree *gp_library;
    int     n_gp_library;
    double (*eval_fn)(void*, double, double, double, double);

    /* Learned bank (plant → controller cache) */
    DsoBankEntry bank[DSO_BANK_MAX];
    int          n_bank;

    /* Resource profiles per tier */
    int cycles[4];     /* GP, LQR, PID, LeadLag */
    int ram[4];
    int branch[4];

    /* Statistics */
    int deploy_count;
    int cache_hits;
} DsoConfig;

/* Default DSO config */
void dso_config_init(DsoConfig *cfg, GpTree *gp_library, int n_gp,
                     double (*eval_fn)(void*,double,double,double,double));

/* Run DSO: fingerprint → cache lookup → tiered selection → deploy */
ControllerResult dso_simulate(DsoConfig *cfg,
                               const Plant *plant, int steps, double dt,
                               uint64_t *seed);

/* Learn from result: update bank entry for this fingerprint */
void dso_learn(DsoConfig *cfg, DsoFingerprint fp, const ControllerResult *result);

/* ================================================================
 * GP controller (tree evaluator wrapper)
 * ================================================================ */
ControllerResult gp_simulate(void *tree, const Plant *plant, int steps, double dt, uint64_t *seed,
                              double (*eval_fn)(void*,double,double,double,double));

#ifdef __cplusplus
}
#endif

#endif /* DSO_CONTROLLERS_H */
