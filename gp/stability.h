#ifndef DSO_STABILITY_H
#define DSO_STABILITY_H

#include "plant.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Stability analysis results */
typedef struct {
    int    n_worlds;         /* total worlds tested */
    int    n_stable;         /* finite IAE, no divergence */
    int    n_oscillatory;    /* oscillation detected (IAE last 25% > 30% of total) */
    int    n_settled;        /* final error < 0.05 */
    double stable_rate;      /* n_stable / n_worlds */
    double worst_iae;        /* max IAE across stable worlds */
    double worst_oscillation_ratio; /* max (last_25pct_IAE / total_IAE) */
    double gain_margin;      /* estimated gain margin (multiplier before instability) */
} StabilityReport;

/* Run stability analysis on a GP controller.
 *
 * n_worlds       — number of random plants to test
 * steps          — simulation steps per world (longer = better stability detection)
 * dt             — timestep
 * seed           — RNG seed
 * perturb_range  — fraction to perturb plant params (e.g. 0.3 = ±30%)
 * tree           — GP tree (passed to eval_fn)
 * eval_fn        — tree evaluator
 *
 * Returns stability report.
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
