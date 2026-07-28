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
 * DSO = best PID from bank (matches Python DSO plan logic)
 * ================================================================ */

/* PID candidate bank: 4 kp × 3 ki × 3 kd = 36 candidates */
static const double dso_kp_list[] = {0.8, 1.3, 2.0, 2.9};
static const double dso_ki_list[] = {0.0, 0.12, 0.28};
static const double dso_kd_list[] = {0.0, 0.08, 0.20};
#define DSO_N_KP 4
#define DSO_N_KI 3
#define DSO_N_KD 3

ControllerResult dso_simulate(const Plant *plant, int steps, double dt,
                               uint64_t *seed)
{
    /* Try all 36 PID candidates, pick best score */
    double best_score = 1e100;
    ControllerResult best = {0};

    double total_time = steps * dt;
    for (int ip = 0; ip < DSO_N_KP; ip++) {
        for (int ii = 0; ii < DSO_N_KI; ii++) {
            for (int id = 0; id < DSO_N_KD; id++) {
                double kp = dso_kp_list[ip];
                double ki = dso_ki_list[ii];
                double kd = dso_kd_list[id];
                ControllerResult r = pid_simulate(kp, ki, kd, plant, steps, dt, seed);
                double wcet, jitter;
                controller_resource_metrics(32, 36, 0, &wcet, &jitter);
                double sc = controller_score(r.itae, total_time, r.overshoot, r.energy,
                                              r.settling_time, wcet, jitter);
                if (sc < best_score) {
                    best_score = sc;
                    best = r;
                }
            }
        }
    }

    /* DSO uses fixed resource profile */
    best.cycles = 32;
    best.ram_bytes = 36;
    best.branch_points = 0;
    double total_time2 = steps * dt;
    controller_resource_metrics(32, 36, 0, &best.wcet_us, &best.jitter_us);
    best.score = controller_score(best.itae, total_time2, best.overshoot, best.energy,
                                   best.settling_time, best.wcet_us, best.jitter_us);
    return best;
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
