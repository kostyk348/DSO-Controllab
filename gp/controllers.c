#include "controllers.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * Internal: simulate a plant with given controller function
 *
 * controller_fn(state, target, y, dt) → u
 * ================================================================ */
typedef struct {
    double target;
    double integral;
    double prev_error;
    double last_y;       /* for LQR/MPC velocity estimation */
} ControllerState;

static ControllerResult simulate_with(
    const Plant *plant, int steps, double dt, uint64_t *seed,
    void *ctrl_ctx,
    double (*compute)(void *ctx, ControllerState *st, double y, double dt))
{
    double x[2] = {0, 0}, y = 0;
    ControllerState st;
    st.target = 1.0;
    st.integral = 0.0;
    st.prev_error = 0.0;
    st.last_y = 0.0;

    double iae = 0, itae = 0, overshoot = 0, energy = 0;
    int saturated = 0;
    double total_time = steps * dt;
    double settling_time = total_time;  /* last time |error| > 2% */
    int has_settled = 0;

    for (int s = 0; s < steps; s++) {
        double t = s * dt;
        double u = compute(ctrl_ctx, &st, y, dt);
        if (u >= 3.999 || u <= -3.999) saturated++;

        double x_next[2];
        plant_step(plant, x, u, dt, x_next, &y, seed);
        x[0] = x_next[0];
        x[1] = x_next[1];

        double error = fabs(st.target - y);
        iae += error * dt;
        itae += error * t * dt;          /* time-weighted IAE */
        energy += u * u;
        if (y - st.target > overshoot) overshoot = y - st.target;

        /* Settling detection: error within ±2% of target */
        if (!has_settled) {
            if (error < 0.02 * st.target) {
                has_settled = 1;
                settling_time = t;
            }
        } else if (error > 0.02 * st.target) {
            has_settled = 0;  /* un-settle if kicked out */
            settling_time = total_time;
        }
    }

    ControllerResult r;
    r.iae = iae;
    r.itae = itae;
    r.overshoot = overshoot > 0 ? overshoot : 0;
    r.energy = energy / steps;
    r.settling_time = has_settled ? settling_time : total_time;
    r.saturated = saturated;
    return r;
}

/* ================================================================
 * PID
 * ================================================================ */

typedef struct {
    double kp, ki, kd;
} PidCtx;

static double pid_compute(void *ctx, ControllerState *st, double y, double dt) {
    PidCtx *p = (PidCtx*)ctx;
    double error = st->target - y;
    st->integral += error * dt;
    st->integral = clamp_val(st->integral, -8.0, 8.0);
    double deriv = (error - st->prev_error) / dt;
    st->prev_error = error;
    double u = p->kp * error + p->ki * st->integral + p->kd * deriv;
    return clamp_val(u, -4.0, 4.0);
}

ControllerResult pid_simulate(double kp, double ki, double kd,
                               const Plant *plant, int steps, double dt,
                               uint64_t *seed)
{
    PidCtx ctx = {kp, ki, kd};
    return simulate_with(plant, steps, dt, seed, &ctx, pid_compute);
}

/* ================================================================
 * LQR
 * ================================================================ */

void lqr_compute_gain(const Plant *plant, double dt, double K[2])
{
    /* Compute A and B matrices */
    double A[2][2], B[2];
    plant_matrices(plant, dt, A, B);

    /* Q = diag(12, 1), R = 0.08 */
    double Q[2][2] = {{12.0, 0.0}, {0.0, 1.0}};
    double R = 0.08;

    /* P = Q (initial), Bm = B as column vector */
    double P[2][2] = {{12.0, 0.0}, {0.0, 1.0}};
    double Bm[2][1] = {{B[0]}, {B[1]}};

    /* DARE iteration: up to 160 steps */
    for (int iter = 0; iter < 160; iter++) {
        /* S = R + Bm^T @ P @ Bm (scalar) */
        double BtPB = Bm[0][0] * (P[0][0] * Bm[0][0] + P[0][1] * Bm[1][0])
                    + Bm[1][0] * (P[1][0] * Bm[0][0] + P[1][1] * Bm[1][0]);
        double S = R + BtPB;

        /* gain = solve(S, Bm^T @ P @ A)   — since S is 1x1, this is just division */
        /* tmp = Bm^T @ P @ A  (1x2) */
        double tmp[2]; /* row vector 1x2 */
        /* tmp = Bm^T @ (P @ A) */
        double PA[2][2];
        PA[0][0] = P[0][0]*A[0][0] + P[0][1]*A[1][0];
        PA[0][1] = P[0][0]*A[0][1] + P[0][1]*A[1][1];
        PA[1][0] = P[1][0]*A[0][0] + P[1][1]*A[1][0];
        PA[1][1] = P[1][0]*A[0][1] + P[1][1]*A[1][1];
        tmp[0] = Bm[0][0]*PA[0][0] + Bm[1][0]*PA[1][0];
        tmp[1] = Bm[0][0]*PA[0][1] + Bm[1][0]*PA[1][1];

        double gain[2]; /* 1x2 */
        gain[0] = tmp[0] / S;
        gain[1] = tmp[1] / S;

        /* P_next = A^T @ P @ A - A^T @ P @ Bm @ gain + Q */
        /* A^T @ P @ A */
        double AtP[2][2];
        AtP[0][0] = A[0][0]*P[0][0] + A[1][0]*P[1][0];
        AtP[0][1] = A[0][0]*P[0][1] + A[1][0]*P[1][1];
        AtP[1][0] = A[0][1]*P[0][0] + A[1][1]*P[1][0];
        AtP[1][1] = A[0][1]*P[0][1] + A[1][1]*P[1][1];

        double AtPA[2][2];
        AtPA[0][0] = AtP[0][0]*A[0][0] + AtP[0][1]*A[1][0];
        AtPA[0][1] = AtP[0][0]*A[0][1] + AtP[0][1]*A[1][1];
        AtPA[1][0] = AtP[1][0]*A[0][0] + AtP[1][1]*A[1][0];
        AtPA[1][1] = AtP[1][0]*A[0][1] + AtP[1][1]*A[1][1];

        /* A^T @ P @ Bm (2x1) */
        double AtPB[2];
        AtPB[0] = AtP[0][0]*Bm[0][0] + AtP[0][1]*Bm[1][0];
        AtPB[1] = AtP[1][0]*Bm[0][0] + AtP[1][1]*Bm[1][0];

        /* (AtPB) @ gain (2x1 @ 1x2 = 2x2) */
        double correction[2][2];
        correction[0][0] = AtPB[0] * gain[0];
        correction[0][1] = AtPB[0] * gain[1];
        correction[1][0] = AtPB[1] * gain[0];
        correction[1][1] = AtPB[1] * gain[1];

        double P_next[2][2];
        P_next[0][0] = AtPA[0][0] - correction[0][0] + Q[0][0];
        P_next[0][1] = AtPA[0][1] - correction[0][1] + Q[0][1];
        P_next[1][0] = AtPA[1][0] - correction[1][0] + Q[1][0];
        P_next[1][1] = AtPA[1][1] - correction[1][1] + Q[1][1];

        /* Check convergence */
        double diff = 0;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                if (fabs(P_next[i][j] - P[i][j]) > diff)
                    diff = fabs(P_next[i][j] - P[i][j]);

        P[0][0] = P_next[0][0]; P[0][1] = P_next[0][1];
        P[1][0] = P_next[1][0]; P[1][1] = P_next[1][1];

        if (diff < 1e-9) break;
    }

    /* K = solve(R + Bm^T @ P @ Bm, Bm^T @ P @ A) */
    double BtPB_final = Bm[0][0] * (P[0][0] * Bm[0][0] + P[0][1] * Bm[1][0])
                      + Bm[1][0] * (P[1][0] * Bm[0][0] + P[1][1] * Bm[1][0]);
    double S_final = R + BtPB_final;

    /* PA final */
    double PAf[2][2];
    PAf[0][0] = P[0][0]*A[0][0] + P[0][1]*A[1][0];
    PAf[0][1] = P[0][0]*A[0][1] + P[0][1]*A[1][1];
    PAf[1][0] = P[1][0]*A[0][0] + P[1][1]*A[1][0];
    PAf[1][1] = P[1][0]*A[0][1] + P[1][1]*A[1][1];

    double BtPA[2];
    BtPA[0] = Bm[0][0]*PAf[0][0] + Bm[1][0]*PAf[1][0];
    BtPA[1] = Bm[0][0]*PAf[0][1] + Bm[1][0]*PAf[1][1];

    K[0] = BtPA[0] / S_final;
    K[1] = BtPA[1] / S_final;
}

typedef struct {
    double K[2];
} LqrCtx;

static double lqr_compute(void *ctx, ControllerState *st, double y, double dt) {
    LqrCtx *lqr = (LqrCtx*)ctx;
    double velocity = (y - st->last_y) / dt;
    st->last_y = y;
    double xhat0 = y - st->target;
    double xhat1 = velocity;
    double u = -(lqr->K[0] * xhat0 + lqr->K[1] * xhat1);
    return clamp_val(u, -4.0, 4.0);
}

ControllerResult lqr_simulate(const double K[2],
                               const Plant *plant, int steps, double dt,
                               uint64_t *seed)
{
    LqrCtx ctx;
    ctx.K[0] = K[0];
    ctx.K[1] = K[1];
    return simulate_with(plant, steps, dt, seed, &ctx, lqr_compute);
}

/* ================================================================
 * MPC
 * ================================================================ */

typedef struct {
    const Plant *plant;
    double dt;
} MpcCtx;

static double mpc_compute(void *ctx, ControllerState *st, double y, double dt) {
    MpcCtx *m = (MpcCtx*)ctx;
    double velocity = (y - st->last_y) / dt;
    st->last_y = y;

    double state[2] = {y, velocity};
    double best_u = 0.0;
    double best_cost = 1e100;

    /* 13 candidates: linspace(-3.0, 3.0, 13) */
    int n_candidates = 13;
    for (int i = 0; i < n_candidates; i++) {
        double u = -3.0 + i * (6.0 / (n_candidates - 1));

        double x_pred[2] = {state[0], state[1]};
        double cost = 0.0;

        for (int h = 0; h < 8; h++) {
            double y_pred;
            double x_next[2];
            /* Predict step: NO rng (deterministic) */
            plant_step(m->plant, x_pred, u, m->dt, x_next, &y_pred, NULL);
            x_pred[0] = x_next[0];
            x_pred[1] = x_next[1];
            cost += (st->target - y_pred) * (st->target - y_pred) + 0.015 * u * u;
        }

        if (cost < best_cost) {
            best_cost = cost;
            best_u = u;
        }
    }

    return clamp_val(best_u, -4.0, 4.0);
}

ControllerResult mpc_simulate(const Plant *plant, int steps, double dt,
                               uint64_t *seed)
{
    MpcCtx ctx;
    ctx.plant = plant;
    ctx.dt = dt;
    return simulate_with(plant, steps, dt, seed, &ctx, mpc_compute);
}

/* ================================================================
 * Lead-Lag compensator (ultra-light: 2 cycles, 0 branches)
 *
 *   C(s) = k * (s + z) / (s + p)
 *   Tustin: u[k] = b0*e[k] + b1*e[k-1] - a1*u[k-1]
 * ================================================================ */

typedef struct {
    double b0, b1, a1;  /* difference equation coefficients */
    double e_prev, u_prev;
} LeadLagCtx;

static void leadlag_init(LeadLagCtx *ll, double k, double z, double p, double dt) {
    /* Tustin discretization of k*(s+z)/(s+p) */
    double denom = 2.0 + p * dt;
    ll->b0 = k * (2.0 + z * dt) / denom;
    ll->b1 = k * (z * dt - 2.0) / denom;
    ll->a1 = (p * dt - 2.0) / denom;
    ll->e_prev = 0;
    ll->u_prev = 0;
}

static double leadlag_compute(void *ctx, ControllerState *st, double y, double dt) {
    (void)dt;
    LeadLagCtx *ll = (LeadLagCtx*)ctx;
    double error = st->target - y;
    double u = ll->b0 * error + ll->b1 * ll->e_prev - ll->a1 * ll->u_prev;
    ll->e_prev = error;
    ll->u_prev = u;
    return clamp_val(u, -4.0, 4.0);
}

ControllerResult leadlag_simulate(double k, double z, double p,
                                   const Plant *plant, int steps, double dt,
                                   uint64_t *seed)
{
    LeadLagCtx ll;
    leadlag_init(&ll, k, z, p, dt);
    ControllerResult r = simulate_with(plant, steps, dt, seed, &ll, leadlag_compute);
    r.controller_tier = 3;
    return r;
}

/* ================================================================
 * DSO — DataSpace OS Controller Runtime
 *
 * Multi-tier adaptive controller with plant fingerprinting
 * and online learning from past deployments.
 *
 * Tiers (lowest resource first):
 *   0: GP library  (evolved trees, 0 branch points)
 *   1: LQR         (computed via DARE)
 *   2: PID         (best-of-36 from candidate bank)
 *   3: Lead-Lag    (2 cycles, failsafe)
 * ================================================================ */

/* ─── PID candidate bank ──────────────────────────────────────── */
static const double dso_kp_list[] = {0.8, 1.3, 2.0, 2.9};
static const double dso_ki_list[] = {0.0, 0.12, 0.28};
static const double dso_kd_list[] = {0.0, 0.08, 0.20};
#define DSO_N_KP 4
#define DSO_N_KI 3
#define DSO_N_KD 3

/* ─── Default resource profiles per tier ──────────────────────── */
static const int default_cycles[4]  = {42, 58, 42, 2};
static const int default_ram[4]     = {40, 64, 40, 8};
static const int default_branch[4]  = {1,  1,  1,  0};

/* ─── Plant fingerprint ──────────────────────────────────────────
 * Quantize plant parameters and hash to 64 bits.
 * Two plants with similar dynamics get the SAME fingerprint.
 * ──────────────────────────────────────────────────────────────── */
DsoFingerprint dso_fingerprint(const Plant *p) {
    /* Quantize: map continuous params to discrete bins */
    int wn_bin    = (int)((p->wn    - 0.5) / 0.5);      /* 0.5–5.0 → bins */
    int zeta_bin  = (int)((p->zeta  - 0.1) / 0.15);     /* 0.1–2.0 → bins */
    int gain_bin  = (int)((p->gain  - 0.1) / 0.3);      /* 0.1–3.0 → bins */
    int delay_bin = p->delay_steps;

    if (wn_bin   < 0)   wn_bin   = 0;
    if (wn_bin   > 15)  wn_bin   = 15;
    if (zeta_bin < 0)   zeta_bin = 0;
    if (zeta_bin > 15)  zeta_bin = 15;
    if (gain_bin < 0)   gain_bin = 0;
    if (gain_bin > 15)  gain_bin = 15;
    if (delay_bin < 0)  delay_bin = 0;
    if (delay_bin > 15) delay_bin = 15;

    /* Pack into 64-bit: 16 bits each */
    return ((DsoFingerprint)wn_bin   << 48)
         | ((DsoFingerprint)zeta_bin << 32)
         | ((DsoFingerprint)gain_bin << 16)
         | ((DsoFingerprint)delay_bin);
}

/* ─── Default config ──────────────────────────────────────────── */
void dso_config_init(DsoConfig *cfg, GpTree *gp_library, int n_gp,
                     double (*eval_fn)(void*,double,double,double,double))
{
    memset(cfg, 0, sizeof(*cfg));
    cfg->gp_library   = gp_library;
    cfg->n_gp_library = n_gp;
    cfg->eval_fn      = eval_fn;
    for (int i = 0; i < 4; i++) {
        cfg->cycles[i] = default_cycles[i];
        cfg->ram[i]    = default_ram[i];
        cfg->branch[i] = default_branch[i];
    }
}

/* ─── Internal: run a specific tier ───────────────────────────── */
static ControllerResult dso_run_tier(DsoConfig *cfg, int tier,
                                      const Plant *plant, int steps, double dt,
                                      uint64_t *seed,
                                      int gp_idx, const double K_lqr[2],
                                      double kp, double ki, double kd,
                                      double k_lead, double z_lead, double p_lead)
{
    ControllerResult r;
    memset(&r, 0, sizeof(r));

    switch (tier) {
        case 0: { /* GP library */
            if (gp_idx < 0 || gp_idx >= cfg->n_gp_library) {
                memset(&r, 0, sizeof(r));
                r.score = 1e10;
                return r;
            }
            r = gp_simulate((void*)&cfg->gp_library[gp_idx], plant, steps, dt, seed, cfg->eval_fn);
            break;
        }
        case 1: { /* LQR */
            double K[2] = {K_lqr[0], K_lqr[1]};
            r = lqr_simulate(K, plant, steps, dt, seed);
            break;
        }
        case 2: { /* PID — best-of-36 */
            double best_sc = 1e100;
            double total_time = steps * dt;
            for (int ip = 0; ip < DSO_N_KP; ip++) {
                for (int ii = 0; ii < DSO_N_KI; ii++) {
                    for (int id = 0; id < DSO_N_KD; id++) {
                        ControllerResult cr = pid_simulate(
                            dso_kp_list[ip], dso_ki_list[ii], dso_kd_list[id],
                            plant, steps, dt, seed);
                        double wcet, jitter;
                        controller_resource_metrics(cfg->cycles[tier], cfg->ram[tier],
                                                     cfg->branch[tier], &wcet, &jitter);
                        double sc = controller_score(cr.itae, total_time,
                                                      cr.overshoot, cr.energy,
                                                      cr.settling_time, wcet, jitter);
                        if (sc < best_sc) {
                            best_sc = sc;
                            r = cr;
                            kp = dso_kp_list[ip];
                            ki = dso_ki_list[ii];
                            kd = dso_kd_list[id];
                        }
                    }
                }
            }
            break;
        }
        case 3: { /* Lead-Lag */
            r = leadlag_simulate(k_lead, z_lead, p_lead, plant, steps, dt, seed);
            break;
        }
        default:
            r.score = 1e10;
            return r;
    }

    /* Apply resource metrics */
    double wcet, jitter;
    controller_resource_metrics(cfg->cycles[tier], cfg->ram[tier],
                                 cfg->branch[tier], &wcet, &jitter);
    r.wcet_us = wcet;
    r.jitter_us = jitter;
    r.cycles = cfg->cycles[tier];
    r.ram_bytes = cfg->ram[tier];
    r.branch_points = cfg->branch[tier];
    r.controller_tier = tier;

    double total_time = steps * dt;
    r.score = controller_score(r.itae, total_time, r.overshoot, r.energy,
                                r.settling_time, r.wcet_us, r.jitter_us);
    return r;
}

/* ─── DSO simulate — main entry ──────────────────────────────────
 *
 * 1. Fingerprint the plant
 * 2. Check bank cache → if found, try cached tier first
 * 3. If miss or cache fails → try tiers 0→3 in order
 * 4. Pick first tier that passes contract
 * 5. Learn: update bank entry
 * ──────────────────────────────────────────────────────────────── */
ControllerResult dso_simulate(DsoConfig *cfg,
                               const Plant *plant, int steps, double dt,
                               uint64_t *seed)
{
    DsoFingerprint fp = dso_fingerprint(plant);
    cfg->deploy_count++;

    /* Contract (matching Python DeploymentContract) */
    double total_time = steps * dt;
    double iae_max = 5.0;
    double overshoot_max = 1.0;
    double sat_frac_max = 0.45;
    double final_error_tol = 1.25;

    /* ── Step 1: Look up cache ───────────────────────────────── */
    DsoBankEntry *cached = NULL;
    for (int i = 0; i < cfg->n_bank; i++) {
        if (cfg->bank[i].fp == fp) {
            cached = &cfg->bank[i];
            cfg->cache_hits++;
            break;
        }
    }

    int start_tier = 0;
    int gp_idx = -1;
    double K_lqr[2] = {0, 0};
    double kp = 0, ki = 0, kd = 0;
    double k_lead = 1.0, z_lead = 5.0, p_lead = 10.0;

    if (cached) {
        start_tier = cached->tier;
        gp_idx = cached->gp_idx;
        K_lqr[0] = cached->K_lqr[0];
        K_lqr[1] = cached->K_lqr[1];
        kp = cached->kp; ki = cached->ki; kd = cached->kd;
        k_lead = cached->k_lead; z_lead = cached->z_lead; p_lead = cached->p_lead;
    }

    /* ── Step 2: Try tiers from start_tier upward ────────────── */
    ControllerResult best;
    memset(&best, 0, sizeof(best));
    best.score = 1e10;

    for (int tier = start_tier; tier < 4; tier++) {
        int try_gp_idx = (tier == 0) ? gp_idx : -1;
        if (tier == 0 && try_gp_idx < 0) {
            /* No cached GP — try all GP trees in library */
            for (int gi = 0; gi < cfg->n_gp_library; gi++) {
                ControllerResult r = dso_run_tier(cfg, 0, plant, steps, dt,
                                                   seed, gi, NULL, 0,0,0, 0,0,0);
                if (r.score < best.score) {
                    best = r;
                    gp_idx = gi;
                }
            }
            if (best.score < 1e9) {
                /* Check basic contract */
                if (best.iae < iae_max &&
                    best.overshoot < overshoot_max &&
                    (double)best.saturated / steps < sat_frac_max) {
                    goto done;
                }
            }
            continue;
        }

        /* Compute LQR gain on-demand */
        if (tier == 1 && K_lqr[0] == 0 && K_lqr[1] == 0) {
            lqr_compute_gain(plant, dt, K_lqr);
        }

        ControllerResult r = dso_run_tier(cfg, tier, plant, steps, dt,
                                           seed, try_gp_idx, K_lqr,
                                           kp, ki, kd, k_lead, z_lead, p_lead);

        /* Accept if this tier's result is the best so far */
        if (r.score < best.score) {
            best = r;
        }

        /* Check contract: if passes, deploy this tier */
        if (r.iae < iae_max &&
            r.overshoot < overshoot_max &&
            (double)r.saturated / steps < sat_frac_max &&
            r.settling_time < total_time * 0.95) {
            best = r;
            goto done;
        }
    }

done:
    /* ── Step 3: Learn ───────────────────────────────────────── */
    if (cached) {
        cached->hit_count++;
        if (best.score < cached->best_score || cached->best_score < 0.1) {
            cached->best_score = best.score;
            cached->tier = best.controller_tier;
        }
    } else if (cfg->n_bank < DSO_BANK_MAX) {
        DsoBankEntry *e = &cfg->bank[cfg->n_bank++];
        e->fp = fp;
        e->tier = best.controller_tier;
        e->gp_idx = (best.controller_tier == 0) ? gp_idx : -1;
        if (best.controller_tier == 1) { e->K_lqr[0] = K_lqr[0]; e->K_lqr[1] = K_lqr[1]; }
        e->kp = kp; e->ki = ki; e->kd = kd;
        e->k_lead = k_lead; e->z_lead = z_lead; e->p_lead = p_lead;
        e->best_score = best.score;
        e->hit_count = 1;
    }

    return best;
}

/* ─── Learn (external call to update bank with known result) ──── */
void dso_learn(DsoConfig *cfg, DsoFingerprint fp, const ControllerResult *result)
{
    for (int i = 0; i < cfg->n_bank; i++) {
        if (cfg->bank[i].fp == fp) {
            cfg->bank[i].hit_count++;
            if (result->score < cfg->bank[i].best_score || cfg->bank[i].best_score < 0.1) {
                cfg->bank[i].best_score = result->score;
                cfg->bank[i].tier = result->controller_tier;
            }
            return;
        }
    }
    if (cfg->n_bank < DSO_BANK_MAX) {
        DsoBankEntry *e = &cfg->bank[cfg->n_bank++];
        e->fp = fp;
        e->tier = result->controller_tier;
        e->best_score = result->score;
        e->hit_count = 1;
    }
}

/* ================================================================
 * GP
 * ================================================================ */

typedef struct {
    void *tree;
    double (*eval_fn)(void*, double, double, double, double);
} GpCtx;

static double gp_compute(void *ctx, ControllerState *st, double y, double dt) {
    GpCtx *g = (GpCtx*)ctx;
    double error = st->target - y;
    st->integral += error * dt;
    double deriv = (error - st->prev_error) / dt;
    st->prev_error = error;
    return g->eval_fn(g->tree, error, st->integral, deriv, y);
}

ControllerResult gp_simulate(void *tree, const Plant *plant, int steps, double dt, uint64_t *seed,
                              double (*eval_fn)(void*, double, double, double, double))
{
    GpCtx ctx;
    ctx.tree = tree;
    ctx.eval_fn = eval_fn;
    return simulate_with(plant, steps, dt, seed, &ctx, gp_compute);
}
