#include "plant.h"
#include <math.h>

static inline uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}
static inline double runif(uint64_t *s, double lo, double hi) {
    return lo + ((splitmix64(s) >> 11) * 0x1.0p-53) * (hi - lo);
}
static inline double rnormal(uint64_t *s, double mean, double std) {
    double u = runif(s, 0, 1), v = runif(s, 0, 1);
    if (u < 1e-15) u = 1e-15;
    return mean + std * sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
}

Plant plant_random(uint64_t *rng) {
    Plant p;
    p.wn          = runif(rng, 0.6, 2.8);
    p.zeta        = runif(rng, 0.12, 1.4);
    p.gain        = runif(rng, 0.55, 1.8);
    p.delay_steps = (int)runif(rng, 0, 8);
    p.disturbance = rnormal(rng, 0.0, 0.04);
    p.noise       = runif(rng, 0.0, 0.015);
    return p;
}

void plant_matrices(const Plant *p, double dt, double A[2][2], double B[2]) {
    double wn2 = p->wn * p->wn;
    A[0][0] = 1.0;              A[0][1] = dt;
    A[1][0] = -wn2 * dt;        A[1][1] = 1.0 - 2.0 * p->zeta * p->wn * dt;
    B[0] = 0.0;                 B[1] = p->gain * dt;
}

void plant_step(const Plant *p, const double x[2], double u, double dt,
                double x_next[2], double *y, uint64_t *rng) {
    double A[2][2], B[2];
    plant_matrices(p, dt, A, B);
    x_next[0] = A[0][0]*x[0] + A[0][1]*x[1] + B[0]*u;
    x_next[1] = A[1][0]*x[0] + A[1][1]*x[1] + B[1]*u + p->disturbance * dt;
    *y = x_next[0];
    if (rng && p->noise > 0.0)
        *y += rnormal(rng, 0.0, p->noise);
}
