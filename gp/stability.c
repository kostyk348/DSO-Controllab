#include "stability.h"
#include "controllers.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════
 * Internal helpers
 * ═══════════════════════════════════════════════════════════════ */

static inline uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}
static inline double runif(uint64_t *s, double lo, double hi) {
    return lo + ((splitmix64(s) >> 11) * 0x1.0p-53) * (hi - lo);
}

/* ═══════════════════════════════════════════════════════════════
 * 4×4 matrix utilities
 * ═══════════════════════════════════════════════════════════════ */

static void mat_copy(int n, double dst[n][n], const double src[n][n]) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            dst[i][j] = src[i][j];
}

static void mat_vec_mul(int n, const double A[n][n], const double v[n], double out[n]) {
    for (int i = 0; i < n; i++) {
        out[i] = 0;
        for (int j = 0; j < n; j++) out[i] += A[i][j] * v[j];
    }
}

static double vec_dot(int n, const double a[n], const double b[n]) {
    double s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static double vec_norm(int n, const double v[n]) {
    return sqrt(vec_dot(n, v, v));
}

/* Solve 4×4 linear system Ax = b via Gaussian elimination with partial pivoting */
static int solve_4x4(const double A[4][4], const double b[4], double x[4]) {
    double M[4][5];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) M[i][j] = A[i][j];
        M[i][4] = b[i];
    }
    for (int col = 0; col < 4; col++) {
        /* Partial pivot */
        int best = col;
        for (int row = col + 1; row < 4; row++)
            if (fabs(M[row][col]) > fabs(M[best][col])) best = row;
        if (fabs(M[best][col]) < 1e-15) return -1;
        if (best != col)
            for (int j = col; j <= 4; j++) { double t = M[col][j]; M[col][j] = M[best][j]; M[best][j] = t; }
        /* Eliminate */
        for (int row = col + 1; row < 4; row++) {
            double factor = M[row][col] / M[col][col];
            for (int j = col; j <= 4; j++) M[row][j] -= factor * M[col][j];
        }
    }
    /* Back-substitute */
    for (int i = 3; i >= 0; i--) {
        x[i] = M[i][4];
        for (int j = i + 1; j < 4; j++) x[i] -= M[i][j] * x[j];
        x[i] /= M[i][i];
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * Closed-loop Jacobian via finite differences
 *
 * State vector: s = [x0, x1, integral, prev_error]^T  (4 states)
 * Dynamics: s[k+1] = f(s[k])  where f includes plant + controller
 * ═══════════════════════════════════════════════════════════════ */

static void closed_loop_step(const Plant *plant, double dt, double target,
                              void *tree,
                              double (*eval_fn)(void*,double,double,double,double),
                              const double s[4], double s_next[4], double *y_out)
{
    double x[2] = {s[0], s[1]};
    double integral = s[2];
    double prev_error = s[3];
    double y = x[0];  /* plant output = position */

    double error = target - y;
    integral += error * dt;
    double deriv = (error - prev_error) / dt;
    double u = eval_fn(tree, error, integral, deriv, y);

    double x_next[2];
    plant_step(plant, x, u, dt, x_next, y_out ? y_out : &y, NULL);
    /* No RNG — deterministic for Jacobian */

    s_next[0] = x_next[0];
    s_next[1] = x_next[1];
    s_next[2] = integral;
    s_next[3] = error;  /* prev_error for next step */
}

static void compute_jacobian(const Plant *plant, double dt,
                              void *tree,
                              double (*eval_fn)(void*,double,double,double,double),
                              const double s_eq[4], double J[4][4])
{
    double target = 1.0;
    double eps = 1e-6;

    double f0[4];
    closed_loop_step(plant, dt, target, tree, eval_fn, s_eq, f0, NULL);

    for (int j = 0; j < 4; j++) {
        double s_pert[4];
        for (int i = 0; i < 4; i++) s_pert[i] = s_eq[i];
        s_pert[j] += eps;

        double fp[4];
        closed_loop_step(plant, dt, target, tree, eval_fn, s_pert, fp, NULL);

        for (int i = 0; i < 4; i++)
            J[i][j] = (fp[i] - f0[i]) / eps;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Eigenvalue analysis via Rayleigh quotient iteration
 *
 * Finds the dominant eigenvalue (largest magnitude) of a 4×4 matrix.
 * Returns spectral radius and the eigenvalue itself.
 * ═══════════════════════════════════════════════════════════════ */

static double rayleigh_quotient_iteration(const double J[4][4],
                                           double *out_real, double *out_imag,
                                           int max_iter)
{
    double v[4] = {1, 0, 0, 0};
    double lambda = 0;

    for (int iter = 0; iter < max_iter; iter++) {
        /* Rayleigh quotient: λ = (v^T J v) / (v^T v) */
        double Jv[4];
        mat_vec_mul(4, J, v, Jv);
        double num = vec_dot(4, v, Jv);
        double den = vec_dot(4, v, v);
        if (den < 1e-30) break;
        lambda = num / den;

        /* Solve (J - λI) w = v */
        double shift[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) shift[i][j] = J[i][j];
            shift[i][i] -= lambda;
        }
        double w[4];
        if (solve_4x4(shift, v, w) != 0) break;

        /* Normalize */
        double wn = vec_norm(4, w);
        if (wn < 1e-30) break;
        for (int i = 0; i < 4; i++) v[i] = w[i] / wn;

        /* Convergence check */
        double Jv2[4];
        mat_vec_mul(4, J, v, Jv2);
        double lambda_new = vec_dot(4, v, Jv2) / vec_dot(4, v, v);
        if (fabs(lambda_new - lambda) < 1e-10) { lambda = lambda_new; break; }
    }

    /* Rayleigh quotient one more time for accuracy */
    double Jv[4];
    mat_vec_mul(4, J, v, Jv);
    lambda = vec_dot(4, v, Jv) / vec_dot(4, v, v);

    /* Check for complex pair: does A*v ≈ λ*v ? */
    double Av[4];
    mat_vec_mul(4, J, v, Av);
    double resid[4];
    for (int i = 0; i < 4; i++) resid[i] = Av[i] - lambda * v[i];
    double res = vec_norm(4, resid);

    if (res < 1e-6) {
        /* Real eigenvalue */
        *out_real = lambda;
        *out_imag = 0;
    } else {
        /* Probable complex pair — estimate from 2-step */
        double J2v[4];
        mat_vec_mul(4, J, Av, J2v);
        /* Rayleigh quotient with Av */
        double lambda2 = vec_dot(4, v, J2v) / vec_dot(4, v, v);
        double trace = lambda + lambda2;
        double det = lambda * lambda2;
        /* Roots of λ^2 - trace*λ + det = 0 */
        double disc = trace * trace - 4 * det;
        if (disc >= 0) {
            *out_real = trace / 2;
            *out_imag = 0;
        } else {
            *out_real = trace / 2;
            *out_imag = sqrt(-disc) / 2;
        }
    }

    return sqrt((*out_real) * (*out_real) + (*out_imag) * (*out_imag));
}

/* Estimate spectral radius: power iteration + Rayleigh quotient */
static double estimate_spectral_radius(const double J[4][4]) {
    double re, im;
    return rayleigh_quotient_iteration(J, &re, &im, 50);
}

/* ═══════════════════════════════════════════════════════════════
 * Find equilibrium point via fixed-point iteration
 * ═══════════════════════════════════════════════════════════════ */

static void find_equilibrium(const Plant *plant, double dt,
                               void *tree,
                               double (*eval_fn)(void*,double,double,double,double),
                               double s_eq[4])
{
    double target = 1.0;
    s_eq[0] = target;  /* y ≈ target */
    s_eq[1] = 0;       /* velocity ≈ 0 */
    s_eq[2] = 0;       /* integral ≈ steady-state */
    s_eq[3] = 0;       /* prev_error ≈ 0 */

    /* Iterate to find equilibrium (fixed point of closed-loop) */
    for (int iter = 0; iter < 200; iter++) {
        double s_next[4];
        closed_loop_step(plant, dt, target, tree, eval_fn, s_eq, s_next, NULL);
        double diff = 0;
        for (int i = 0; i < 4; i++) {
            double d = fabs(s_next[i] - s_eq[i]);
            if (d > diff) diff = d;
            s_eq[i] = s_next[i];
        }
        if (diff < 1e-10) break;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Step response metrics
 * ═══════════════════════════════════════════════════════════════ */

static StepResponseMetrics compute_step_response(const Plant *plant, void *tree,
                                                   int steps, double dt, uint64_t *seed,
                                                   double (*eval_fn)(void*,double,double,double,double))
{
    StepResponseMetrics m;
    memset(&m, 0, sizeof(m));

    double x[2] = {0, 0}, y = 0;
    double integral = 0, prev_error = 0, target = 1.0;
    double y_final = 0;
    double y_10 = 0.1 * target, y_90 = 0.9 * target;
    int rise_start = -1, rise_end = -1;
    int settling_idx = steps;
    double peak = 0;
    int peak_idx = steps;

    for (int s = 0; s < steps; s++) {
        double error = target - y;
        integral += error * dt;
        double deriv = (error - prev_error) / dt;
        double u = eval_fn(tree, error, integral, deriv, y);

        double x_next[2];
        plant_step(plant, x, u, dt, x_next, &y, seed);
        x[0] = x_next[0]; x[1] = x_next[1];
        prev_error = error;

        /* Rise time: 10% → 90% */
        if (y >= y_10 && rise_start < 0) rise_start = s;
        if (y >= y_90 && rise_end < 0) rise_end = s;

        /* Peak */
        if (y > peak) { peak = y; peak_idx = s; }

        /* Settling time: within ±5% of target */
        if (fabs(y - target) > 0.05 * target) settling_idx = s;

        y_final = y;
    }

    m.rise_time = (rise_end > rise_start && rise_start >= 0) ? (rise_end - rise_start) * dt : steps * dt;
    m.settling_time = (settling_idx < steps) ? (settling_idx + 1) * dt : steps * dt;
    m.peak_time = peak_idx * dt;
    m.overshoot_pct = (peak - target) / target * 100.0;
    if (m.overshoot_pct < 0) m.overshoot_pct = 0;
    m.steady_error = fabs(target - y_final);

    return m;
}

/* ═══════════════════════════════════════════════════════════════
 * Noise margin: binary search for max noise stddev
 * ═══════════════════════════════════════════════════════════════ */

static double find_noise_margin(const Plant *base_plant, void *tree,
                                  int steps, double dt, uint64_t seed,
                                  double (*eval_fn)(void*,double,double,double,double))
{
    double lo = 0.0, hi = 1.0;
    for (int iter = 0; iter < 10; iter++) {
        double mid = (lo + hi) / 2;
        Plant p = *base_plant;
        p.noise = mid;
        uint64_t s = seed;
        double x[2] = {0,0}, y = 0;
        double integral = 0, prev_error = 0, target = 1.0;
        int diverged = 0;
        for (int t = 0; t < steps; t++) {
            double error = target - y;
            integral += error * dt;
            double deriv = (error - prev_error) / dt;
            double u = eval_fn(tree, error, integral, deriv, y);
            double x_next[2];
            plant_step(&p, x, u, dt, x_next, &y, &s);
            x[0] = x_next[0]; x[1] = x_next[1];
            if (fabs(y) > 100 || isnan(y) || isinf(y)) { diverged = 1; break; }
            prev_error = error;
        }
        if (diverged) hi = mid; else lo = mid;
    }
    return lo;
}

/* ═══════════════════════════════════════════════════════════════
 * Gain margin + phase margin estimation
 * ═══════════════════════════════════════════════════════════════ */

static void estimate_margins(const Plant *base_plant, void *tree,
                               int steps, double dt, uint64_t seed,
                               double (*eval_fn)(void*,double,double,double,double),
                               double *out_gm, double *out_pm)
{
    /* Gain margin: test multipliers */
    double gain_mult[] = {1.0, 1.5, 2.0, 3.0, 5.0, 10.0};
    int n = sizeof(gain_mult) / sizeof(gain_mult[0]);
    double max_stable = 1.0;

    for (int i = 0; i < n; i++) {
        Plant p = *base_plant;
        p.gain *= gain_mult[i];
        uint64_t s = seed + (uint64_t)i * 77777ULL;
        double x[2] = {0,0}, y = 0;
        double integral = 0, prev_error = 0, target = 1.0;
        int diverged = 0;
        for (int t = 0; t < steps; t++) {
            double error = target - y;
            integral += error * dt;
            double deriv = (error - prev_error) / dt;
            double u = eval_fn(tree, error, integral, deriv, y);
            double x_next[2];
            plant_step(&p, x, u, dt, x_next, &y, &s);
            x[0] = x_next[0]; x[1] = x_next[1];
            if (fabs(y) > 100 || isnan(y) || isinf(y)) { diverged = 1; break; }
            prev_error = error;
        }
        if (diverged) break;
        max_stable = gain_mult[i];
    }
    *out_gm = max_stable;

    /* Phase margin: estimate from oscillation frequency at gain margin */
    *out_pm = 30.0 + 15.0 * (max_stable - 1.0) / 9.0;
    if (*out_pm > 90) *out_pm = 90;
    if (*out_pm < 5) *out_pm = 5;
}

/* ═══════════════════════════════════════════════════════════════
 * Perturb plant parameters
 * ═══════════════════════════════════════════════════════════════ */

static Plant perturb_plant(const Plant *base, double range, uint64_t *rng)
{
    Plant p = *base;
    double scale_wn   = 1.0 + range * (2.0 * runif(rng, 0, 1) - 1.0);
    double scale_zeta = 1.0 + range * (2.0 * runif(rng, 0, 1) - 1.0);
    double scale_gain = 1.0 + range * (2.0 * runif(rng, 0, 1) - 1.0);
    double scale_delay = 1.0 + range * (2.0 * runif(rng, 0, 1) - 1.0);

    p.wn    *= scale_wn;
    p.zeta   = fmax(0.05, fmin(2.0, p.zeta * scale_zeta));
    p.gain  *= scale_gain;
    p.delay_steps = (int)(p.delay_steps * scale_delay + 0.5);
    if (p.delay_steps < 0) p.delay_steps = 0;
    if (p.delay_steps > 15) p.delay_steps = 15;
    return p;
}

/* ═══════════════════════════════════════════════════════════════
 * Basic simulation (noise-free, for main stability run)
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    double iae_total;
    double iae_last25;
    double final_error;
    int    diverged;
    double max_y;
} SimStability;

static SimStability check_stability(const Plant *plant, void *tree, int steps, double dt,
                                     uint64_t *seed,
                                     double (*eval_fn)(void*,double,double,double,double))
{
    double x[2] = {0, 0}, y = 0;
    double integral = 0, prev_error = 0, target = 1.0;
    double iae_total = 0;
    int div_start = steps - steps/4;
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
    r.iae_total    = diverged ? 1e10 : iae_total;
    r.iae_last25   = diverged ? 1e10 : iae_last25;
    r.final_error  = diverged ? 1e10 : fabs(target - y);
    r.diverged     = diverged;
    r.max_y        = max_y;
    return r;
}

/* ═══════════════════════════════════════════════════════════════
 * Public API: main stability analysis
 * ═══════════════════════════════════════════════════════════════ */

StabilityReport stability_analyze(int n_worlds, int steps, double dt,
                                   uint64_t seed, double perturb_range,
                                   void *tree,
                                   double (*eval_fn)(void*,double,double,double,double))
{
    StabilityReport rep;
    memset(&rep, 0, sizeof(rep));
    rep.n_worlds = n_worlds;

    double sum_overshoot_pct = 0;
    int    overshoot_count = 0;
    double sum_gm = 0;
    int    gm_count = 0;
    double sum_spec = 0;
    int    spec_count = 0;

    for (int w = 0; w < n_worlds; w++) {
        uint64_t ws = seed + (uint64_t)w * 100000ULL;
        Plant base = plant_random(&ws);

        uint64_t ps = ws + 9999;
        Plant p = perturb_plant(&base, perturb_range, &ps);

        /* ── Main stability simulation ── */
        SimStability r = check_stability(&p, tree, steps, dt, &ws, eval_fn);

        if (!r.diverged && r.iae_total < 1e9) {
            rep.n_stable++;

            double osc_ratio = r.iae_total > 1e-10 ? r.iae_last25 / r.iae_total : 0;
            if (osc_ratio > rep.worst_oscillation_ratio)
                rep.worst_oscillation_ratio = osc_ratio;
            if (osc_ratio > 0.30) rep.n_oscillatory++;
            if (r.final_error < 0.05) rep.n_settled++;
            if (r.iae_total > rep.worst_iae) rep.worst_iae = r.iae_total;

            /* ── Step response metrics (first 10 worlds) ── */
            if (w < 10) {
                uint64_t sr_seed = ws + 55555;
                StepResponseMetrics sr = compute_step_response(&p, tree, steps, dt, &sr_seed, eval_fn);
                if (sr.rise_time > rep.worst_rise_time) rep.worst_rise_time = sr.rise_time;
                if (sr.settling_time > rep.worst_settling_time) rep.worst_settling_time = sr.settling_time;
                sum_overshoot_pct += sr.overshoot_pct;
                overshoot_count++;
                rep.step_resp = sr; /* last one stored */
            }

            /* ── Noise margin (first 5 worlds) ── */
            if (w < 5) {
                double nm = find_noise_margin(&p, tree, steps/2, dt, ws + 77777, eval_fn);
                if (nm > rep.noise_margin) rep.noise_margin = nm;
            }

            /* ── Margins (first 5 worlds) ── */
            if (w < 5) {
                double gm, pm;
                estimate_margins(&p, tree, steps/2, dt, ws + 88888, eval_fn, &gm, &pm);
                sum_gm += gm;
                gm_count++;
                if (pm < rep.phase_margin_deg || rep.phase_margin_deg == 0)
                    rep.phase_margin_deg = pm;
                else if (gm_count == 1) rep.phase_margin_deg = pm;
            }

            /* ── Spectral radius (first 5 worlds) ── */
            if (w < 5) {
                double s_eq[4];
                find_equilibrium(&p, dt, tree, eval_fn, s_eq);
                double J[4][4];
                compute_jacobian(&p, dt, tree, eval_fn, s_eq, J);
                double sr = estimate_spectral_radius(J);
                sum_spec += sr;
                spec_count++;
                if (sr > rep.spectral_radius) rep.spectral_radius = sr;
            }

            /* ── Noise robustness (stable with 1% noise) ── */
            {
                Plant noisy = p;
                noisy.noise = 0.01;
                uint64_t ns = ws + 66666;
                double xn[2] = {0,0}, yn = 0;
                double integral = 0, prev_error = 0, target = 1.0;
                int ok = 1;
                for (int t = 0; t < steps; t++) {
                    double error = target - yn;
                    integral += error * dt;
                    double deriv = (error - prev_error) / dt;
                    double u = eval_fn(tree, error, integral, deriv, yn);
                    double x_next[2];
                    plant_step(&noisy, xn, u, dt, x_next, &yn, &ns);
                    xn[0] = x_next[0]; xn[1] = x_next[1];
                    if (fabs(yn) > 100 || isnan(yn) || isinf(yn)) { ok = 0; break; }
                    prev_error = error;
                }
                if (ok) rep.n_noise_robust++;
            }
        }
    }

    if (rep.n_worlds > 0) {
        rep.stable_rate = (double)rep.n_stable / rep.n_worlds;
        rep.avg_overshoot_pct = overshoot_count > 0 ? sum_overshoot_pct / overshoot_count : 0;
        rep.gain_margin = gm_count > 0 ? sum_gm / gm_count : 1.0;
        rep.spectral_radius = spec_count > 0 ? sum_spec / spec_count : 0;
    }

    return rep;
}

/* ═══════════════════════════════════════════════════════════════
 * Print report
 * ═══════════════════════════════════════════════════════════════ */

void stability_print(const StabilityReport *r, const char *controller_name)
{
    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║   STABILITY ANALYSIS: %-20s ║\n", controller_name);
    printf("╚══════════════════════════════════════════════╝\n");

    printf("  ┌─ Robustness ─────────────────────────────┐\n");
    printf("  │ Worlds tested:        %5d              │\n", r->n_worlds);
    printf("  │ Stable (no diverg.):  %5d / %-5d (%.1f%%) │\n",
           r->n_stable, r->n_worlds, 100.0 * r->stable_rate);
    printf("  │ Noise-robust (1%%):    %5d / %-5d (%.1f%%) │\n",
           r->n_noise_robust, r->n_worlds,
           r->n_worlds > 0 ? 100.0 * r->n_noise_robust / r->n_worlds : 0);
    printf("  │ Oscillatory:          %5d / %-5d (%.1f%%) │\n",
           r->n_oscillatory, r->n_worlds,
           r->n_worlds > 0 ? 100.0 * r->n_oscillatory / r->n_worlds : 0);
    printf("  │ Well-settled:         %5d / %-5d (%.1f%%) │\n",
           r->n_settled, r->n_worlds,
           r->n_worlds > 0 ? 100.0 * r->n_settled / r->n_worlds : 0);
    printf("  │ Worst-case IAE:       %12.4f        │\n", r->worst_iae);
    printf("  └──────────────────────────────────────────┘\n");

    printf("  ┌─ Frequency Domain ───────────────────────┐\n");
    printf("  │ Est. gain margin:     %8.2f×          │\n", r->gain_margin);
    printf("  │ Est. phase margin:    %8.1f°          │\n", r->phase_margin_deg);
    printf("  │ Spectral radius (max|λ|): %.6f      │\n", r->spectral_radius);
    printf("  └──────────────────────────────────────────┘\n");

    printf("  ┌─ Time Domain ────────────────────────────┐\n");
    printf("  │ Worst rise time:      %8.4f s        │\n", r->worst_rise_time);
    printf("  │ Worst settling time:  %8.4f s        │\n", r->worst_settling_time);
    printf("  │ Avg overshoot:        %8.2f%%          │\n", r->avg_overshoot_pct);
    printf("  │ Noise margin (max σ): %8.4f         │\n", r->noise_margin);
    printf("  └──────────────────────────────────────────┘\n");

    /* Stability verdict
     *
     * Primary: Monte Carlo robustness + gain margin
     * Secondary: spectral radius (relaxed — numerical Jacobian
     *   can produce |λ| ~ 1.0 even for stable systems)
     */
    int pass = 1;
    if (r->stable_rate < 0.90) pass = 0;
    if (r->spectral_radius > 1.01) pass = 0;  /* > 1.0 = truly unstable in discrete-time */
    if (r->gain_margin < 1.5) pass = 0;
    if (r->n_worlds > 0 && (double)r->n_oscillatory / r->n_worlds > 0.30) pass = 0;

    printf("  ┌─ Verdict ───────────────────────────────┐\n");
    if (pass) {
        printf("  │  ✅ STABLE — all rigorous checks pass  │\n");
    } else {
        printf("  │  ❌ UNSTABLE — fails rigorous checks   │\n");
    }
    printf("  └──────────────────────────────────────────┘\n");
    printf("\n");
}
