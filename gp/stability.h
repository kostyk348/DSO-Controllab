#ifndef DSO_STABILITY_H
#define DSO_STABILITY_H

#include "plant.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Step response metrics */
typedef struct {
    double rise_time;       /* 10% → 90% rise time (seconds) */
    double settling_time;   /* settle within ±5% (seconds) */
    double peak_time;       /* time to first peak (seconds) */
    double overshoot_pct;   /* percentage overshoot */
    double steady_error;    /* final steady-state error */
} StepResponseMetrics;

/* Stability analysis results */
typedef struct {
    int    n_worlds;             /* total worlds tested */
    int    n_stable;             /* finite IAE, no divergence */
    int    n_oscillatory;        /* IAE last 25% > 30% of total */
    int    n_settled;            /* final error < 0.05 */
    int    n_noise_robust;       /* stable with 1% measurement noise */
    double stable_rate;          /* n_stable / n_worlds */
    double worst_iae;            /* max IAE across stable worlds */
    double worst_oscillation_ratio;
    double gain_margin;          /* gain multiplier before instability */
    double phase_margin_deg;     /* estimated phase margin (degrees) */
    double spectral_radius;      /* max |eigenvalue| of linearized sys */
    double noise_margin;         /* max noise stddev before divergence */
    double worst_rise_time;      /* worst-case rise time (s) */
    double worst_settling_time;  /* worst-case settling time (s) */
    double avg_overshoot_pct;    /* average overshoot percentage */
    StepResponseMetrics step_resp;
} StabilityReport;

/* Run stability analysis on a GP controller.
 *
 * n_worlds       — number of random plants to test
 * steps          — simulation steps per world
 * dt             — timestep
 * seed           — RNG seed
 * perturb_range  — fraction to perturb plant params (e.g. 0.3 = ±30%)
 * tree           — GP tree (passed to eval_fn)
 * eval_fn        — tree evaluator
 */
StabilityReport stability_analyze(int n_worlds, int steps, double dt,
                                   uint64_t seed, double perturb_range,
                                   void *tree,
                                   double (*eval_fn)(void*,double,double,double,double));

/* Print stability report to stdout */
void stability_print(const StabilityReport *r, const char *controller_name);

#ifdef __cplusplus
}
#endif

#endif /* DSO_STABILITY_H */
