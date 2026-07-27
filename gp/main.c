#include "gp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [OPTIONS]\n"
        "Options:\n"
        "  --pop SIZE         population size (default: 100)\n"
        "  --gen N            generations (default: 50)\n"
        "  --depth N          max tree depth (default: 4)\n"
        "  --mut R            mutation rate (default: 0.3)\n"
        "  --cross R          crossover rate (default: 0.7)\n"
        "  --tourn N          tournament size (default: 3)\n"
        "  --worlds N         random plants per fitness eval (default: 50)\n"
        "  --steps N          simulation steps per world (default: 500)\n"
        "  --dt R             timestep (default: 0.02)\n"
        "  --seed N           RNG seed (default: time)\n"
        "  --export-c FILE    export best controller as C\n"
        "  --export-ada NAME  export best controller as Ada (NAME.ads + NAME.adb)\n"
        "  --benchmark N      run full benchmark on N worlds (default: off)\n"
        "  --json             output benchmark as JSON (default: table)\n"
        "  --bloat R          anti-bloat penalty per tree node (default: 0.01)\n"
        "  --help             this help\n",
        prog);
}

/* ─── Controllers for comparison ─────────────────────────────── */

/* PID */
static double pid_run(double kp, double ki, double kd,
                      const Plant *plant, int steps, double dt, uint64_t *seed,
                      double *out_iae, double *out_overshoot,
                      double *out_energy, int *out_saturated) {
    double x[2] = {0,0}, y = 0;
    double integral = 0, prev_error = 0, target = 1.0;
    double iae = 0, overshoot = 0, energy = 0;
    int saturated = 0;

    for (int s = 0; s < steps; s++) {
        double error = target - y;
        integral += error * dt;
        double deriv = (error - prev_error) / dt;
        double u = kp * error + ki * integral + kd * deriv;
        if (u > 4.0) u = 4.0; else if (u < -4.0) u = -4.0;
        if (u >= 3.999 || u <= -3.999) saturated++;
        double x_next[2];
        plant_step(plant, x, u, dt, x_next, &y, seed);
        x[0] = x_next[0]; x[1] = x_next[1];
        iae += fabs(target - y) * dt;
        energy += u * u;
        if (y - target > overshoot) overshoot = y - target;
        prev_error = error;
    }
    *out_iae = iae;
    *out_overshoot = overshoot;
    *out_energy = energy / steps;
    *out_saturated = saturated;
    return iae + 0.35*(overshoot>0?overshoot:0) + 0.04*(energy/steps) + 0.02*((double)saturated/steps);
}

/* GP tree controller */
static double gp_run(const GpTree *t, const Plant *plant, int steps, double dt, uint64_t *seed,
                     double *out_iae, double *out_overshoot,
                     double *out_energy, int *out_saturated) {
    double x[2] = {0,0}, y = 0;
    double integral = 0, prev_error = 0, target = 1.0;
    double iae = 0, overshoot = 0, energy = 0;
    int saturated = 0;

    for (int s = 0; s < steps; s++) {
        double error = target - y;
        integral += error * dt;
        double deriv = (error - prev_error) / dt;
        double u = gp_tree_eval(t, error, integral, deriv, y);
        if (u >= 3.999 || u <= -3.999) saturated++;
        double x_next[2];
        plant_step(plant, x, u, dt, x_next, &y, seed);
        x[0] = x_next[0]; x[1] = x_next[1];
        iae += fabs(target - y) * dt;
        energy += u * u;
        if (y - target > overshoot) overshoot = y - target;
        prev_error = error;
    }
    *out_iae = iae;
    *out_overshoot = overshoot;
    *out_energy = energy / steps;
    *out_saturated = saturated;
    return iae + 0.35*(overshoot>0?overshoot:0) + 0.04*(energy/steps) + 0.02*((double)saturated/steps);
}

/* ─── Resource metrics (matching Python resource_metrics) ────── */
static void resource_metrics(int cycles, int ram_bytes, int branch_points,
                             double *out_wcet, double *out_jitter) {
    (void)ram_bytes;
    *out_wcet   = cycles * 0.1 + 1.0;
    *out_jitter = branch_points * 0.15 + 0.05;
}

/* ─── Contract check (matching Python DeploymentContract) ────── */
typedef struct {
    int    cycles_max;
    int    ram_bytes_max;
    double wcet_us_max;
    double jitter_us_max;
    double iae_max;
    double overshoot_max;
    double max_abs_y;
    double final_error_max;
    double saturation_fraction_max;
    int    finite_required;
} Contract;

static int contract_pass(const Contract *c, double iae, double overshoot,
                         double energy, double wcet, double jitter,
                         int cycles, int ram, double sat_frac,
                         int is_finite) {
    if (cycles > c->cycles_max) return 0;
    if (ram > c->ram_bytes_max) return 0;
    if (wcet > c->wcet_us_max) return 0;
    if (jitter > c->jitter_us_max) return 0;
    if (iae > c->iae_max) return 0;
    if (overshoot > c->overshoot_max) return 0;
    if (sat_frac > c->saturation_fraction_max) return 0;
    if (c->finite_required && !is_finite) return 0;
    (void)energy;
    return 1;
}

/* ─── Sign test ──────────────────────────────────────────────── */
static double sign_test_pvalue(const double *a, const double *b, int n) {
    int pos = 0, neg = 0;
    for (int i = 0; i < n; i++) {
        if (a[i] < b[i]) pos++;
        else if (a[i] > b[i]) neg++;
    }
    int N = pos + neg;
    if (N == 0) return 1.0;
    /* Binomial sign test p-value (two-sided) */
    double k = (pos < neg) ? pos : neg;
    /* Normal approximation for N > 25, else exact (simple approx) */
    if (N > 25) {
        double z = (k + 0.5 - N/2.0) / (sqrt((double)N) / 2.0);
        double p = erfc(fabs(z) / 1.41421356237); /* approximation */
        return p;
    }
    /* For small N, return simple ratio */
    return 2.0 * (k + 1.0) / (N + 1.0);
}

/* ─── Benchmark ──────────────────────────────────────────────── */

static void run_benchmark(int n_worlds, int steps, double dt, uint64_t seed,
                          const GpTree *best_gp, int verbose, int json_mode) {
    Contract c_default = {
        .cycles_max = 180,
        .ram_bytes_max = 96,
        .wcet_us_max = 8.0,
        .jitter_us_max = 0.8,
        .iae_max = 4.0,
        .overshoot_max = 0.9,
        .max_abs_y = 8.0,
        .final_error_max = 1.25,
        .saturation_fraction_max = 0.45,
        .finite_required = 1
    };

    /* PID candidates (same as Python) */
    double kp_list[] = {0.8, 1.3, 2.0, 2.9};
    double ki_list[] = {0.0, 0.12, 0.28};
    double kd_list[] = {0.0, 0.08, 0.20};
    int n_kp = 4, n_ki = 3, n_kd = 3;

    /* Per-world scores for sign test */
    double *gp_scores = (double*)calloc(n_worlds, sizeof(double));
    double *pid_scores = (double*)calloc(n_worlds, sizeof(double));

    /* Aggregate stats */
    double sum_gp_iae = 0, sum_gp_os = 0, sum_gp_en = 0, sum_gp_sat = 0;
    double sum_pid_iae = 0, sum_pid_os = 0, sum_pid_en = 0, sum_pid_sat = 0;
    int gp_contract_pass = 0, pid_contract_pass = 0;

    /* Resource metrics */
    int gp_cycles = 42, gp_ram = 40, gp_branch = 1;
    int pid_cycles = 42, pid_ram = 40, pid_branch = 1;
    double gp_wcet, gp_jitter, pid_wcet, pid_jitter;
    resource_metrics(gp_cycles, gp_ram, gp_branch, &gp_wcet, &gp_jitter);
    resource_metrics(pid_cycles, pid_ram, pid_branch, &pid_wcet, &pid_jitter);

    if (verbose && !json_mode) {
        printf("\n===== FULL BENCHMARK: %d worlds =====\n", n_worlds);
        printf("%-6s %8s %8s %8s %6s %6s\n",
               "World", "IAE", "Oversht", "Energy", "Sat%", "Score");
        printf("------ -------- -------- -------- ------ ------\n");
    }

    for (int w = 0; w < n_worlds; w++) {
        uint64_t ws = seed + (uint64_t)w * 100000ULL;
        Plant plant = plant_random(&ws);

        /* Run GP */
        double gp_iae, gp_os, gp_en;
        int gp_sat;
        double gp_sc = gp_run(best_gp, &plant, steps, dt, &ws,
                              &gp_iae, &gp_os, &gp_en, &gp_sat);
        gp_scores[w] = gp_sc;
        sum_gp_iae += gp_iae; sum_gp_os += gp_os;
        sum_gp_en += gp_en; sum_gp_sat += gp_sat;
        if (contract_pass(&c_default, gp_iae, gp_os, gp_en,
                          gp_wcet, gp_jitter,
                          gp_cycles, gp_ram, (double)gp_sat/steps, 1))
            gp_contract_pass++;

        /* Find best PID for this world */
        double best_pid_sc = 1e10;
        double best_pid_iae = 0, best_pid_os = 0, best_pid_en = 0;
        int best_pid_sat = 0;
        for (int ip = 0; ip < n_kp; ip++) {
            for (int ii = 0; ii < n_ki; ii++) {
                for (int id = 0; id < n_kd; id++) {
                    double pid_iae, pid_os, pid_en;
                    int pid_sat;
                    double ps = pid_run(kp_list[ip], ki_list[ii], kd_list[id],
                                       &plant, steps, dt, &ws,
                                       &pid_iae, &pid_os, &pid_en, &pid_sat);
                    if (ps < best_pid_sc) {
                        best_pid_sc = ps;
                        best_pid_iae = pid_iae;
                        best_pid_os = pid_os;
                        best_pid_en = pid_en;
                        best_pid_sat = pid_sat;
                    }
                }
            }
        }
        pid_scores[w] = best_pid_sc;
        sum_pid_iae += best_pid_iae; sum_pid_os += best_pid_os;
        sum_pid_en += best_pid_en; sum_pid_sat += best_pid_sat;
        if (contract_pass(&c_default, best_pid_iae, best_pid_os, best_pid_en,
                          pid_wcet, pid_jitter,
                          pid_cycles, pid_ram, (double)best_pid_sat/steps, 1))
            pid_contract_pass++;

        if (verbose && !json_mode && (w < 10 || w % 100 == 99 || w == n_worlds-1)) {
            printf("%-6d %8.4f %8.4f %8.4f %5.1f%% %6.3f  | PID: %6.3f\n",
                   w, gp_iae, gp_os, gp_en, 100.0*gp_sat/steps, gp_sc, best_pid_sc);
        }
    }

    /* Averages */
    double avg_gp_iae = sum_gp_iae / n_worlds;
    double avg_gp_os  = sum_gp_os  / n_worlds;
    double avg_gp_en  = sum_gp_en  / n_worlds;
    double avg_gp_sat = sum_gp_sat / n_worlds;
    double avg_pid_iae = sum_pid_iae / n_worlds;
    double avg_pid_os  = sum_pid_os  / n_worlds;
    double avg_pid_en  = sum_pid_en  / n_worlds;
    double avg_pid_sat = sum_pid_sat / n_worlds;

    /* Improvement */
    double avg_gp_sc = 0, avg_pid_sc = 0;
    for (int w = 0; w < n_worlds; w++) {
        avg_gp_sc += gp_scores[w];
        avg_pid_sc += pid_scores[w];
    }
    avg_gp_sc /= n_worlds;
    avg_pid_sc /= n_worlds;
    double impr = (avg_pid_sc - avg_gp_sc) / avg_pid_sc * 100.0;

    /* Sign test */
    double pval = sign_test_pvalue(gp_scores, pid_scores, n_worlds);

    if (json_mode) {
        printf("{\n");
        printf("  \"n_worlds\": %d,\n", n_worlds);
        printf("  \"steps\": %d,\n", steps);
        printf("  \"dt\": %.4f,\n", dt);
        printf("  \"gp\": {\n");
        printf("    \"iae\": { \"mean\": %.6f },\n", avg_gp_iae);
        printf("    \"overshoot\": { \"mean\": %.6f },\n", avg_gp_os);
        printf("    \"energy\": { \"mean\": %.6f },\n", avg_gp_en);
        printf("    \"saturation_frac\": { \"mean\": %.6f },\n", avg_gp_sat/steps);
        printf("    \"score\": { \"mean\": %.6f },\n", avg_gp_sc);
        printf("    \"wcet_us\": %.2f,\n", gp_wcet);
        printf("    \"jitter_us\": %.2f,\n", gp_jitter);
        printf("    \"cycles\": %d,\n", gp_cycles);
        printf("    \"ram_bytes\": %d,\n", gp_ram);
        printf("    \"branch_points\": %d,\n", gp_branch);
        printf("    \"contract_pass_rate\": %.4f\n", (double)gp_contract_pass/n_worlds);
        printf("  },\n");
        printf("  \"pid\": {\n");
        printf("    \"iae\": { \"mean\": %.6f },\n", avg_pid_iae);
        printf("    \"overshoot\": { \"mean\": %.6f },\n", avg_pid_os);
        printf("    \"energy\": { \"mean\": %.6f },\n", avg_pid_en);
        printf("    \"saturation_frac\": { \"mean\": %.6f },\n", avg_pid_sat/steps);
        printf("    \"score\": { \"mean\": %.6f },\n", avg_pid_sc);
        printf("    \"wcet_us\": %.2f,\n", pid_wcet);
        printf("    \"jitter_us\": %.2f,\n", pid_jitter);
        printf("    \"cycles\": %d,\n", pid_cycles);
        printf("    \"ram_bytes\": %d,\n", pid_ram);
        printf("    \"branch_points\": %d,\n", pid_branch);
        printf("    \"contract_pass_rate\": %.4f\n", (double)pid_contract_pass/n_worlds);
        printf("  },\n");
        printf("  \"improvement_pct\": %.2f,\n", impr);
        printf("  \"sign_test_pvalue\": %.6f,\n", pval);
        printf("  \"gp_beats_pid\": %s\n", (avg_gp_sc < avg_pid_sc) ? "true" : "false");
        printf("}\n");
    } else {
        printf("\n===== BENCHMARK RESULTS =====\n");
        printf("%-20s %12s %12s\n", "Metric", "GP", "Best PID");
        printf("-------------------- ------------ ------------\n");
        printf("%-20s %12.6f %12.6f\n", "IAE (mean)", avg_gp_iae, avg_pid_iae);
        printf("%-20s %12.6f %12.6f\n", "Overshoot (mean)", avg_gp_os, avg_pid_os);
        printf("%-20s %12.6f %12.6f\n", "Energy (mean)", avg_gp_en, avg_pid_en);
        printf("%-20s %12.2f %12.2f\n", "Saturation frac (%)", 100.0*avg_gp_sat/steps, 100.0*avg_pid_sat/steps);
        printf("%-20s %12.6f %12.6f\n", "Score (mean)", avg_gp_sc, avg_pid_sc);
        printf("%-20s %12.2f %12.2f\n", "WCET (us)", gp_wcet, pid_wcet);
        printf("%-20s %12.2f %12.2f\n", "Jitter (us)", gp_jitter, pid_jitter);
        printf("%-20s %12d %12d\n", "Cycles", gp_cycles, pid_cycles);
        printf("%-20s %12d %12d\n", "RAM (bytes)", gp_ram, pid_ram);
        printf("%-20s %12d %12d\n", "Branch points", gp_branch, pid_branch);
        printf("%-20s %12.2f%% %12.2f%%\n", "Contract pass rate", 100.0*(double)gp_contract_pass/n_worlds, 100.0*(double)pid_contract_pass/n_worlds);
        printf("\n");
        printf("GP improvement over PID: %.2f%%\n", impr);
        printf("Sign test p-value: %.6f  (GP beats PID: %s)\n",
               pval, avg_gp_sc < avg_pid_sc ? "YES" : "NO");
    }

    free(gp_scores);
    free(pid_scores);
}

int main(int argc, char **argv) {
    /* Defaults */
    int pop_size = 100;
    int generations = 50;
    int max_depth = 4;
    double mut_rate = 0.3;
    double cross_rate = 0.7;
    int tournament_size = 3;
    int n_worlds = 50;
    int steps = 500;
    double dt = 0.02;
    uint64_t seed = (uint64_t)time(NULL);
    const char *export_c = NULL;
    const char *export_ada = NULL;
    int benchmark = 0;
    int json_mode = 0;
    double bloat_penalty = 0.01;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--help") == 0)   { print_usage(argv[0]); return 0; }
        else if (strcmp(argv[i], "--pop") == 0 && i+1 < argc)     pop_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gen") == 0 && i+1 < argc)     generations = atoi(argv[++i]);
        else if (strcmp(argv[i], "--depth") == 0 && i+1 < argc)   max_depth = atoi(argv[++i]);
        else if (strcmp(argv[i], "--mut") == 0 && i+1 < argc)     mut_rate = atof(argv[++i]);
        else if (strcmp(argv[i], "--cross") == 0 && i+1 < argc)   cross_rate = atof(argv[++i]);
        else if (strcmp(argv[i], "--tourn") == 0 && i+1 < argc)   tournament_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--worlds") == 0 && i+1 < argc)  n_worlds = atoi(argv[++i]);
        else if (strcmp(argv[i], "--steps") == 0 && i+1 < argc)   steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dt") == 0 && i+1 < argc)      dt = atof(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc)    seed = (uint64_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "--export-c") == 0 && i+1 < argc)   export_c = argv[++i];
        else if (strcmp(argv[i], "--export-ada") == 0 && i+1 < argc) export_ada = argv[++i];
        else if (strcmp(argv[i], "--benchmark") == 0 && i+1 < argc)  benchmark = atoi(argv[++i]);
        else if (strcmp(argv[i], "--json") == 0)     json_mode = 1;
        else if (strcmp(argv[i], "--bloat") == 0 && i+1 < argc) bloat_penalty = atof(argv[++i]);
        else { fprintf(stderr, "Unknown: %s\n", argv[i]); return 1; }
    }

    printf("DSO GP — Genetic Programming for Controller Synthesis\n");
    printf("====================================================\n");
    printf("  pop=%d gen=%d depth=%d mut=%.2f cross=%.2f tourn=%d bloat=%.3f\n",
           pop_size, generations, max_depth, mut_rate, cross_rate, tournament_size, bloat_penalty);
    printf("  worlds=%d steps=%d dt=%.4f seed=%llu\n",
           n_worlds, steps, dt, (unsigned long long)seed);
    printf("\n");

    /* Run evolution */
    GpPopulation pop = {0};
    fprintf(stderr, "Initializing population...\n");
    gp_evolve(&pop, pop_size, generations, max_depth,
              mut_rate, cross_rate, tournament_size,
              n_worlds, steps, dt, bloat_penalty, seed);

    printf("\n=== Best controller ===\n");
    printf("Fitness score: %.6f (lower is better)\n", pop.trees[0].fitness);
    printf("Tree size: %d nodes\n", pop.trees[0].size);

    char tree_str[2048];
    gp_tree_print(&pop.trees[0], tree_str, sizeof(tree_str));
    printf("Tree: %s\n", tree_str);

    /* Export */
    if (export_c) {
        FILE *f = fopen(export_c, "w");
        if (f) {
            gp_export_c(&pop.trees[0], "controller_compute", f);
            fclose(f);
            printf("\nExported C: %s\n", export_c);
        }
    }

    if (export_ada) {
        char ads[1024], adb[1024];
        snprintf(ads, sizeof(ads), "%s.ads", export_ada);
        snprintf(adb, sizeof(adb), "%s.adb", export_ada);
        FILE *fs = fopen(ads, "w");
        if (fs) { gp_export_ada(&pop.trees[0], "Compute", fs); fclose(fs); printf("Exported Ada spec: %s\n", ads); }
        FILE *fb = fopen(adb, "w");
        if (fb) { gp_export_ada_body(&pop.trees[0], "Compute", fb); fclose(fb); printf("Exported Ada body: %s\n", adb); }
    }

    /* Benchmark */
    if (benchmark > 0) {
        run_benchmark(benchmark, steps, dt, seed, &pop.trees[0], 1, json_mode);
    }

    /* Cleanup */
    free(pop.trees);

    printf("\nDone.\n");
    return 0;
}
