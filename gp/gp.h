#ifndef DSO_GP_H
#define DSO_GP_H

#include <stdio.h>
#include <stdint.h>
#include "plant.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * GP Controller — expression tree in flat array (arena-style)
 *
 * Tree nodes are stored in a compact array for cache efficiency
 * and zero malloc/free during evolution.
 *
 * Terminals available to the controller:
 *   ERROR    = target - y
 *   INTEGRAL = ∫error dt
 *   DERIV    = derror/dt
 *   Y        = current plant output
 *   CONST    = learnable constant
 *
 * Internal nodes: ADD, SUB, MUL, DIV, MIN, MAX, SQ, ABS
 * ================================================================ */

#define GP_MAX_NODES    63   /* enough for depth-5 binary tree */
#define GP_MAX_DEPTH     6

typedef enum {
    /* Binary arithmetic */
    NODE_ADD, NODE_SUB, NODE_MUL, NODE_DIV,
    /* Binary functions */
    NODE_MIN, NODE_MAX,
    /* Unary functions */
    NODE_SQ,  NODE_ABS, NODE_NEG,
    /* Terminals */
    NODE_CONST,
    NODE_ERROR, NODE_INTEGRAL, NODE_DERIV, NODE_Y,
} NodeType;

typedef struct {
    NodeType type;
    double   value;    /* for NODE_CONST */
    int8_t   left;     /* child index in nodes[], -1 = none */
    int8_t   right;
} GpNode;

typedef struct {
    GpNode nodes[GP_MAX_NODES];
    int    size;       /* number of used nodes */
    double fitness;    /* lower = better (set by evaluator) */
} GpTree;

/* ================================================================
 * Population
 * ================================================================ */

typedef struct {
    GpTree *trees;
    int     size;
    int     capacity;
} GpPopulation;

/* ================================================================
 * Random number utilities (xoshiro128+)
 * ================================================================ */

uint64_t gp_rand_u64(uint64_t *s);
double   gp_rand_uniform(uint64_t *s, double lo, double hi);
int      gp_rand_int(uint64_t *s, int lo, int hi);
double   gp_rand_normal(uint64_t *s, double mean, double std);

/* ================================================================
 * Tree construction / manipulation
 * ================================================================ */

/* Generate random tree with given max depth (full or grow method) */
void gp_tree_random(GpTree *t, int max_depth, uint64_t *rng);

/* Deep copy src -> dst */
void gp_tree_copy(GpTree *dst, const GpTree *src);

/* Evaluate tree: given controller state, return control output u */
double gp_tree_eval(const GpTree *t, double error, double integral,
                    double deriv, double y);

/* Point mutation: change a random node's type/constant */
void gp_tree_mutate(GpTree *t, int max_depth, uint64_t *rng);

/* Subtree crossover: swap random subtrees between a and b */
void gp_tree_crossover(GpTree *a, GpTree *b, uint64_t *rng);

/* Print tree to string buffer (for debug / export) */
void gp_tree_print(const GpTree *t, char *buf, int buf_size);

/* ================================================================
 * Fitness evaluation
 * ================================================================ */

/* Simulate tree controller on an ensemble of random plants.
 * Returns combined score (lower = better).
 *
 * Internally runs steps_per_world steps per plant,
 * measures IAE, overshoot, energy, and combines via:
 *   score = iae + 0.35*overshoot + 0.04*energy + 0.02*saturation
 *          + bloat_penalty * tree_size
 *
 * bloat_penalty: anti-bloat coefficient (e.g. 0.01 penalizes
 *   each extra tree node by 0.01 score points)
 */
double gp_fitness(GpTree *t, int n_worlds, int steps, double dt,
                  uint64_t *seed, double bloat_penalty);

/* ================================================================
 * Evolution loop
 * ================================================================ */

/* Run GP evolution.
 *
 * pop         - population (pre-allocated, pop->size = 0 initially)
 * pop_size    - number of individuals
 * generations - number of generations
 * max_depth   - max tree depth for initialization & mutation
 * mut_rate    - per-individual mutation probability (0.0 .. 1.0)
 * cross_rate  - per-pair crossover probability
 * tournament_size - tournament selection size
 * n_worlds    - number of random plants per fitness eval
 * steps       - simulation steps per world
 * dt          - timestep
 * bloat       - anti-bloat penalty per tree node (e.g. 0.01)
 * seed        - RNG seed
 *
 * After run, pop->trees[0] is the best individual.
 */
/* If gen_log is non-NULL, writes CSV: gen,best_fitness,avg_top5 */
void gp_evolve(GpPopulation *pop, int pop_size, int generations,
               int max_depth, double mut_rate, double cross_rate,
               int tournament_size,
               int n_worlds, int steps, double dt,
               double bloat, uint64_t seed, FILE *gen_log);

/* ================================================================
 * Export
 * ================================================================ */

/* Export best tree as C function */
void gp_export_c(const GpTree *t, const char *func_name, FILE *out);

/* Export best tree as Ada SPARK spec (.ads) */
void gp_export_ada(const GpTree *t, const char *func_name, FILE *out);

/* Export best tree as Ada SPARK body (.adb) */
void gp_export_ada_body(const GpTree *t, const char *func_name, FILE *out);

#ifdef __cplusplus
}
#endif

#endif /* DSO_GP_H */
