#ifndef DSO_CONTROLLERS_H
#define DSO_CONTROLLERS_H

#include <stdint.h>
#include "plant.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Controller API
 *
 * All controllers run simulation on a plant, return score.
 * Resource metrics stored by pointer.
 * ================================================================ */

typedef struct {
    double iae;
    double overshoot;
    double energy;         /* mean(u^2) over steps */
    int    saturated;      /* count of steps where |u| >= 3.999 */
    double score;          /* combined: iae + 0.35*os + 0.04*energy + 0.12*wcet + 0.9*jitter */
    double wcet_us;
    double jitter_us;
    int    cycles;
    int    ram_bytes;
    int    branch_points;
} ControllerResult;

/* ─── Resource metrics (matching Python resource_metrics) ────── */
static inline void controller_resource_metrics(int cycles, int ram_bytes, int branch_points,
                                                double *out_wcet, double *out_jitter) {
    (void)ram_bytes;
    *out_wcet   = cycles / 48.0;           /* Python: cycles / 48.0 */
    *out_jitter = 0.04 + 0.055 * branch_points; /* Python: 0.04 + 0.055 * branch_points */
}

/* ─── Score (matching Python score()) ─────────────────────────── */
static inline double controller_score(double iae, double overshoot, double energy,
                                       double wcet_us, double jitter_us) {
    return iae + 0.35 * overshoot + 0.04 * energy + 0.12 * wcet_us + 0.9 * jitter_us;
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

/* Simulate PID with given gains; returns result */
ControllerResult pid_simulate(double kp, double ki, double kd,
                               const Plant *plant, int steps, double dt,
                               uint64_t *seed);

/* ================================================================
 * LQR controller
 * ================================================================ */

/* Compute LQR gain via DARE iteration (Q=diag(12,1), R=0.08) */
void lqr_compute_gain(const Plant *plant, double dt, double K[2]);

/* Simulate LQR controller; returns result */
ControllerResult lqr_simulate(const double K[2],
                               const Plant *plant, int steps, double dt,
                               uint64_t *seed);

/* ================================================================
 * MPC controller
 * ================================================================ */

/* Simulate MPC (horizon=8, 13 candidate u values); returns result */
ControllerResult mpc_simulate(const Plant *plant, int steps, double dt,
                               uint64_t *seed);

/* ================================================================
 * DSO controller (best verified PID from bank)
 * ================================================================ */

/* Find best PID via search over 36 candidates; returns result */
ControllerResult dso_simulate(const Plant *plant, int steps, double dt,
                               uint64_t *seed);

/* ================================================================
 * GP controller
 * ================================================================ */

/* Simulate GP tree controller; returns result */
typedef struct {
    double (*eval)(void *ctx, double error, double integral, double deriv, double y);
    void *ctx;
} GpController;

ControllerResult gp_simulate(void *tree, const Plant *plant, int steps, double dt, uint64_t *seed,
                              double (*eval_fn)(void*,double,double,double,double));

#ifdef __cplusplus
}
#endif

#endif /* DSO_CONTROLLERS_H */
