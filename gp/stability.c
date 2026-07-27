#include "stability.h"
#include "controllers.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ──────────────────────────────────────────────────────────────
 * Run simulation and check for instability/oscillation
 * ────────────────────────────────────────────────────────────── */
typedef struct {
    double iae_total;
    double iae_last25;    /* IAE in last 25% of steps */
    double final_error;
    int    diverged;       /* y > 100 or NaN */
    double max_y;
} SimStability;

static SimStability check_stability(const Plant *plant, void *tree, int steps, double dt,
                                     uint64_t *seed,
                                     double (*eval_fn)(void*,double,double,double,double))
{
    double x[2] = {0, 0}, y = 0;
    double integral = 0, prev_error = 0, target = 1.0;
    double iae_total = 0;
    int div_start = steps - steps/4;  /* last 25% */
    double iae_last25 = 0;
    double max_y = 0;
    int diverged = 0;

    for (int s = 0; s < steps; s++) {
        double error = target - y;
        integral += error * dt;
        double deriv = (error - prev_error) / dt;
        double u = eval_fn(tree, error, integral, deriv, y);

        double x_next[2];
        plant_step(plant, x, u, dt, x_next, &y, seed);
        x[0] = x_next[0]; x[1] = x_next[1];

        double ae = fabs(target - y) * dt;
        iae_total += ae;
        if (s >= div_start) iae_last25 += ae;
        if (fabs(y) > max_y) max_y = fabs(y);
        if (fabs(y) > 100.0 || isnan(y) || isinf(y)) { diverged = 1; break; }
        prev_error = error;
    }

    SimStability r;
    r.iae_total = diverged ? 1e10 : iae_total;
    r.iae_last25 = diverged ? 1e10 : iae_last25;
    r.final_error = diverged ? 1e10 : fabs(target - y);
    r.diverged = diverged;
    r.max_y = max_y;
    return r;
}

/* ──────────────────────────────────────────────────────────────
 * Monte Carlo robustness: perturb plant parameters
 * ────────────────────────────────────────────────────────────── */
static Plant perturb_plant(const Plant *base, double range, uint64_t *rng)
{
    Plant p = *base;
    /* Scale each parameter by random factor in [1-range, 1+range] */
    double scale_wn    = 1.0 + range * (2.0 * ((double)((*rng >> 11) & 0x3fffffff) / 0x40000000) - 1.0);
    double scale_zeta  = 1.0 + range * (2.0 * ((double)((*rng >> 11) & 0x3fffffff) / 0x40000000) - 1.0);
    double scale_gain  = 1.0 + range * (2.0 * ((double)((*rng >> 11) & 0x3fffffff) / 0x40000000) - 1.0);
    double scale_delay = 1.0 + range * (2.0 * ((double)((*rng >> 11) & 0x3fffffff) / 0x40000000) - 1.0);

    p.wn    *= scale_wn;
    p.zeta   = fmax(0.05, fmin(2.0, p.zeta * scale_zeta));
    p.gain  *= scale_gain;
    p.delay_steps = (int)(p.delay_steps * scale_delay + 0.5);
    if (p.delay_steps < 0) p.delay_steps = 0;
    if (p.delay_steps > 15) p.delay_steps = 15;
    return p;
}

/* ──────────────────────────────────────────────────────────────
 * Estimate gain margin by sweeping gain multiplier
 * ────────────────────────────────────────────────────────────── */
static double estimate_gain_margin(const Plant *base_plant, void *tree, int steps, double dt,
                                    uint64_t seed,
                                    double (*eval_fn)(void*,double,double,double,double))
{
    /* Test gain multipliers until instability */
    double gain_mult[] = {1.0, 1.5, 2.0, 3.0, 5.0, 10.0};
    int n = sizeof(gain_mult) / sizeof(gain_mult[0]);
    double max_stable = 1.0;

    for (int i = 0; i < n; i++) {
        Plant p = *base_plant;
        p.gain *= gain_mult[i];
        uint64_t s = seed + (uint64_t)i * 77777ULL;
        SimStability r = check_stability(&p, tree, steps, dt, &s, eval_fn);
        if (r.diverged || r.iae_total > 100.0) break;
        max_stable = gain_mult[i];
    }
    return max_stable;
}

/* ──────────────────────────────────────────────────────────────
 * Public API
 * ────────────────────────────────────────────────────────────── */

StabilityReport stability_analyze(int n_worlds, int steps, double dt,
                                   uint64_t seed, double perturb_range,
                                   void *tree,
                                   double (*eval_fn)(void*,double,double,double,double))
{
    StabilityReport rep;
    memset(&rep, 0, sizeof(rep));
    rep.n_worlds = n_worlds;

    double worst_iae = 0;
    double worst_osc = 0;
    double sum_gain_margin = 0;

    for (int w = 0; w < n_worlds; w++) {
        uint64_t ws = seed + (uint64_t)w * 100000ULL;
        Plant base = plant_random(&ws);

        /* Perturb this plant */
        uint64_t ps = ws + 9999;
        Plant p = perturb_plant(&base, perturb_range, &ps);

        SimStability r = check_stability(&p, tree, steps, dt, &ws, eval_fn);

        if (!r.diverged && r.iae_total < 1e9) {
            rep.n_stable++;
            if (r.iae_total > worst_iae) worst_iae = r.iae_total;

            double osc_ratio = r.iae_total > 1e-10 ? r.iae_last25 / r.iae_total : 0;
            if (osc_ratio > rep.worst_oscillation_ratio)
                rep.worst_oscillation_ratio = osc_ratio;
            if (osc_ratio > 0.30) rep.n_oscillatory++;
            if (r.final_error < 0.05) rep.n_settled++;

            /* Gain margin on first few worlds (expensive) */
            if (w < 5) {
                double gm = estimate_gain_margin(&p, tree, steps/2, dt, ws + 88888, eval_fn);
                sum_gain_margin += gm;
            }
        }
    }

    if (rep.n_worlds > 0) {
        rep.stable_rate = (double)rep.n_stable / rep.n_worlds;
        rep.worst_iae = worst_iae;
        rep.gain_margin = sum_gain_margin / (n_worlds < 5 ? n_worlds : 5);
    }

    return rep;
}

void stability_print(const StabilityReport *r, const char *controller_name)
{
    printf("\n========== STABILITY ANALYSIS: %s ==========\n", controller_name);
    printf("  Worlds tested:         %d\n", r->n_worlds);
    printf("  Stable:                %d / %d (%.1f%%)\n",
           r->n_stable, r->n_worlds, 100.0 * r->stable_rate);
    printf("  Oscillatory:           %d / %d (%.1f%%)\n",
           r->n_oscillatory, r->n_worlds,
           r->n_worlds > 0 ? 100.0 * r->n_oscillatory / r->n_worlds : 0);
    printf("  Well-settled:          %d / %d (%.1f%%)\n",
           r->n_settled, r->n_worlds,
           r->n_worlds > 0 ? 100.0 * r->n_settled / r->n_worlds : 0);
    printf("  Worst-case IAE:        %.4f\n", r->worst_iae);
    printf("  Max oscillation ratio: %.4f\n", r->worst_oscillation_ratio);
    printf("  Est. gain margin:      %.2f×\n", r->gain_margin);
    printf("============================================\n");
}
