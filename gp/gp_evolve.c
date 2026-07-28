#include "gp.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ── RNG (local copies for this TU) ──────────────────────────── */
static inline uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}
static inline double runif(uint64_t *s, double lo, double hi) {
    return lo + ((splitmix64(s) >> 11) * 0x1.0p-53) * (hi - lo);
}
static inline int rint_range(uint64_t *s, int lo, int hi) {
    return lo + (int)(((splitmix64(s) >> 11) * 0x1.0p-53) * (hi - lo + 1));
}

/* ── Fitness evaluation ──────────────────────────────────────── */

double gp_fitness(GpTree *t, int n_worlds, int steps, double dt,
                  uint64_t *seed, double bloat_penalty) {
    double total_score = 0.0;
    int valid_worlds = 0;
    double total_time = steps * dt;

    #pragma omp parallel for reduction(+:total_score, valid_worlds)
    for (int w = 0; w < n_worlds; w++) {
        uint64_t local_seed = *seed + (uint64_t)w * 100000ULL;
        Plant plant = plant_random(&local_seed);

        double x[2] = {0.0, 0.0};
        double y = 0.0;
        double integral = 0.0;
        double prev_error = 0.0;
        double target = 1.0;
        double iae = 0.0;
        double itae = 0.0;
        double overshoot = 0.0;
        double energy = 0.0;
        int saturated = 0;
        double settling_time = total_time;
        int has_settled = 0;

        for (int s = 0; s < steps; s++) {
            double t_sec = s * dt;
            double error = target - y;
            integral += error * dt;
            double deriv = (error - prev_error) / dt;
            double u = gp_tree_eval(t, error, integral, deriv, y);

            if (fabs(u) >= 3.999) saturated++;

            double x_next[2];
            plant_step(&plant, x, u, dt, x_next, &y, &local_seed);
            x[0] = x_next[0]; x[1] = x_next[1];

            double abs_err = fabs(target - y);
            iae += abs_err * dt;
            itae += abs_err * t_sec * dt;   /* ITAE: time-weighted */
            energy += u * u;
            if (y - target > overshoot) overshoot = y - target;

            /* Settling: within ±2% of target */
            if (!has_settled) {
                if (abs_err < 0.02 * target) {
                    has_settled = 1;
                    settling_time = t_sec;
                }
            } else if (abs_err > 0.02 * target) {
                has_settled = 0;  /* kicked out of band */
                settling_time = total_time;
            }
            prev_error = error;
        }

        if (!has_settled) settling_time = total_time;

        /* Energy: average control power */
        double avg_energy = energy / steps;
        /* ITAE normalized: mean time-weighted error */
        double itae_norm = itae / total_time;

        /* Unsettled penalty: fraction of time NOT settled × total_time */
        double unsettled_penalty = 0.005 * (total_time - settling_time);
        if (unsettled_penalty < 0) unsettled_penalty = 0;

        double score = itae_norm                       /* ITAE — time-weighted accuracy */
                     + 0.30 * (overshoot > 0.0 ? overshoot : 0.0)  /* overshoot */
                     + 0.08 * avg_energy               /* control energy (×2 vs old) */
                     + unsettled_penalty                /* settling time penalty */
                     + 0.02 * ((double)saturated / steps)  /* saturation */
                     + bloat_penalty * t->size;         /* anti-bloat */

        if (!isnan(score) && !isinf(score)) {
            total_score += score;
            valid_worlds++;
        }
    }

    if (valid_worlds == 0) return 1e10;
    return total_score / valid_worlds;
}

/* ── Evolution loop ──────────────────────────────────────────── */

void gp_evolve(GpPopulation *pop, int pop_size, int generations,
               int max_depth, double mut_rate, double cross_rate,
               int tournament_size,
               int n_worlds, int steps, double dt,
               double bloat_penalty, uint64_t seed, FILE *gen_log) {

    if (!pop->trees || pop->capacity < pop_size) {
        if (pop->trees) free(pop->trees);
        pop->trees = (GpTree*)calloc((size_t)pop_size, sizeof(GpTree));
        pop->capacity = pop_size;
    }
    pop->size = pop_size;

    uint64_t rng = seed;
    GpTree *trees = pop->trees;

    /* Initialize random population */
    for (int i = 0; i < pop_size; i++) {
        gp_tree_random(&trees[i], max_depth, &rng);
    }

    /* Evaluate initial fitness */
    for (int i = 0; i < pop_size; i++) {
        trees[i].fitness = gp_fitness(&trees[i], n_worlds, steps, dt, &rng, bloat_penalty);
    }

    /* Allocate next generation */
    GpTree *next_gen = (GpTree*)calloc((size_t)pop_size, sizeof(GpTree));

    /* Write gen log header */
    if (gen_log) {
        fprintf(gen_log, "gen,best_fitness,avg_top5\n");
    }

    /* Find initial best */
    GpTree best_all_time;
    int best_idx = 0;
    for (int i = 1; i < pop_size; i++) {
        if (trees[i].fitness < trees[best_idx].fitness) best_idx = i;
    }
    gp_tree_copy(&best_all_time, &trees[best_idx]);

    /* Generation loop */
    for (int gen = 0; gen < generations; gen++) {
        /* ── Adaptive rates ────────────────────────────────
         * Mutation: starts high (exploration), decays to 20% of base
         * Crossover: starts low, ramps up (refinement)
         * ────────────────────────────────────────────────── */
        double progress = (double)gen / generations;
        double cur_mut   = mut_rate   * (1.0 - 0.8 * progress);   /* 100%→20% of base */
        double cur_cross = cross_rate * (0.4 + 0.6 * progress);   /* 40%→100% of base */

        /* Elitism: keep best 2 */
        int elite1 = 0, elite2 = 1;
        if (trees[1].fitness < trees[0].fitness) { elite1 = 1; elite2 = 0; }
        for (int i = 2; i < pop_size; i++) {
            if (trees[i].fitness < trees[elite1].fitness) {
                elite2 = elite1; elite1 = i;
            } else if (trees[i].fitness < trees[elite2].fitness) {
                elite2 = i;
            }
        }
        gp_tree_copy(&next_gen[0], &trees[elite1]);
        gp_tree_copy(&next_gen[1], &trees[elite2]);

        /* Fill rest via tournament + crossover + mutation */
        for (int i = 2; i < pop_size; i++) {
            /* Parent 1 */
            int p1 = rint_range(&rng, 0, pop_size - 1);
            for (int t = 1; t < tournament_size; t++) {
                int c = rint_range(&rng, 0, pop_size - 1);
                if (trees[c].fitness < trees[p1].fitness) p1 = c;
            }
            /* Parent 2 */
            int p2 = rint_range(&rng, 0, pop_size - 1);
            for (int t = 1; t < tournament_size; t++) {
                int c = rint_range(&rng, 0, pop_size - 1);
                if (trees[c].fitness < trees[p2].fitness) p2 = c;
            }

            if (runif(&rng, 0, 1) < cur_cross) {
                gp_tree_copy(&next_gen[i], &trees[p1]);
                GpTree tmp;
                gp_tree_copy(&tmp, &trees[p2]);
                gp_tree_crossover(&next_gen[i], &tmp, &rng);
            } else {
                gp_tree_copy(&next_gen[i], &trees[p1]);
            }

            if (runif(&rng, 0, 1) < cur_mut) {
                gp_tree_mutate(&next_gen[i], max_depth, &rng);
            }
        }

        /* Swap generations */
        for (int i = 0; i < pop_size; i++) {
            gp_tree_copy(&trees[i], &next_gen[i]);
        }

        /* Evaluate fitness */
        #pragma omp parallel for
        for (int i = 0; i < pop_size; i++) {
            trees[i].fitness = gp_fitness(&trees[i], n_worlds, steps, dt, &rng, bloat_penalty);
        }

        /* Track best this generation */
        int gen_best = 0;
        for (int i = 1; i < pop_size; i++) {
            if (trees[i].fitness < trees[gen_best].fitness) gen_best = i;
        }

        if (gen == 0 || trees[gen_best].fitness < best_all_time.fitness) {
            gp_tree_copy(&best_all_time, &trees[gen_best]);
        }

        /* Average of top 5 */
        double avg_top5 = 0;
        for (int k = 0; k < 5 && k < pop_size; k++) avg_top5 += trees[k].fitness;
        avg_top5 /= (5 < pop_size ? 5 : pop_size);

        if (gen_log) {
            fprintf(gen_log, "%d,%.6f,%.6f\n",
                    gen + 1, trees[gen_best].fitness, avg_top5);
            fflush(gen_log);
        }

        fprintf(stderr, "  gen %3d/%d  best=%.4f  avg=%.4f\n",
                gen + 1, generations,
                trees[gen_best].fitness, avg_top5);
    }

    /* Restore all-time best to position 0 */
    gp_tree_copy(&trees[0], &best_all_time);
    trees[0].fitness = gp_fitness(&trees[0], n_worlds, steps, dt, &rng, bloat_penalty);

    free(next_gen);
}
