#ifndef DSO_GP_PLANT_H
#define DSO_GP_PLANT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Second-order plant model
 *
 *   x0' = x1
 *   x1' = -wn^2 * x0 - 2*zeta*wn * x1 + gain * u + disturbance
 *   y   = x0
 *
 * Mirrors Python World in dso_controllab/world.py
 */
typedef struct {
    double wn;           /* natural frequency */
    double zeta;         /* damping ratio */
    double gain;         /* input gain */
    int    delay_steps;  /* input delay in steps (not implemented in step) */
    double disturbance;  /* constant disturbance */
    double noise;        /* measurement noise stddev */
} Plant;

/* Generate random plant with stable dynamics */
Plant plant_random(uint64_t *rng);

/* Compute state-space matrices for given dt */
void plant_matrices(const Plant *p, double dt, double A[2][2], double B[2]);

/* Single step: (x, u, dt) -> (x_next, y)
 *   x[0] = position, x[1] = velocity
 *   u    = control input
 *   dt   = timestep
 *   rng  = NULL -> no measurement noise
 *   Returns y = x_next[0] with optional noise
 */
void plant_step(const Plant *p, const double x[2], double u, double dt,
                double x_next[2], double *y, uint64_t *rng);

#ifdef __cplusplus
}
#endif

#endif /* DSO_GP_PLANT_H */
