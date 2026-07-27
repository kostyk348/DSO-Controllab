#include "gp.h"
#include "controllers.h"
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
        "  --benchmark N      run full 5-way benchmark on N worlds\n"
        "  --sweep N M        multi-seed sweep: seeds N..N+M-1, output CSV\n"
        "  --json             output benchmark as JSON\n"
        "  --bloat R          anti-bloat penalty per tree node (default: 0.01)\n"
        "  --help             this help\n",
        prog);
}

/* ─── Tree eval trampoline ────────────────────────────────────── */
static double tree_eval_wrap(void *ctx, double error, double integral,
                              double deriv, double y) {
    return gp_tree_eval((const GpTree*)ctx, error, integral, deriv, y);
}

/* ─── Contract (matching Python DeploymentContract) ───────────── */
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

static int contract_pass(const Contract *c, const ControllerResult *r,
                          int steps, int cycles, int ram) {
    if (cycles > c->cycles_max) return 0;
    if (ram > c->ram_bytes_max) return 0;
    if (r->wcet_us > c->wcet_us_max) return 0;
    if (r->jitter_us > c->jitter_us_max) return 0;
    if (r->iae > c->iae_max) return 0;
    if (r->overshoot > c->overshoot_max) return 0;
    double sat_frac = (double)r->saturated / steps;
    if (sat_frac > c->saturation_fraction_max) return 0;
    if (c->finite_required) {
        if (!isfinite(r->iae) || !isfinite(r->overshoot) || !isfinite(r->energy))
            return 0;
    }
    return 1;
}

/* ─── Sign test ───────────────────────────────────────────────── */
static double sign_test_pvalue(const double *a, const double *b, int n) {
    int pos = 0, neg = 0;
    for (int i = 0; i < n; i++) {
        if (a[i] < b[i]) pos++;
        else if (a[i] > b[i]) neg++;
    }
    int N = pos + neg;
    if (N == 0) return 1.0;
    double k = (pos < neg) ? (double)pos : (double)neg;
    /* Normal approximation */
    double z = (k + 0.5 - N / 2.0) / (sqrt((double)N) / 2.0);
    return erfc(fabs(z) / 1.41421356237);
}

/* ─── Full 5-way benchmark ────────────────────────────────────── */

typedef struct {
    const char *name;
    int   cycles;
    int   ram;
    int   branch;
} CtrlProfile;

static const CtrlProfile profiles[] = {
    {"GP",   42, 40, 1},
    {"PID",  42, 40, 1},
    {"LQR",  58, 64, 1},
    {"MPC", 620,176,10},
    {"DSO",  32, 36, 0},
};
#define N_CONTROLLERS 5

static void run_benchmark(int n_worlds, int steps, double dt, uint64_t seed,
                          const GpTree *best_gp, int json_mode)
{
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

    /* Per-world scores for sign test */
    double *all_scores[N_CONTROLLERS];
    double *all_iae[N_CONTROLLERS];
    double *all_os[N_CONTROLLERS];
    double *all_en[N_CONTROLLERS];
    int    *all_sat[N_CONTROLLERS];
    int    *all_cpass[N_CONTROLLERS];
    for (int ci = 0; ci < N_CONTROLLERS; ci++) {
        all_scores[ci]  = (double*)calloc(n_worlds, sizeof(double));
        all_iae[ci]     = (double*)calloc(n_worlds, sizeof(double));
        all_os[ci]      = (double*)calloc(n_worlds, sizeof(double));
        all_en[ci]      = (double*)calloc(n_worlds, sizeof(double));
        all_sat[ci]     = (int*)calloc(n_worlds, sizeof(int));
        all_cpass[ci]   = (int*)calloc(n_worlds, sizeof(int));
    }

    double sum_score[N_CONTROLLERS] = {0};

    for (int w = 0; w < n_worlds; w++) {
        uint64_t ws = seed + (uint64_t)w * 100000ULL;
        Plant plant = plant_random(&ws);

        /* Compute LQR gain once per world */
        double K[2];
        lqr_compute_gain(&plant, dt, K);

        /* Run each controller */
        ControllerResult results[N_CONTROLLERS];

        /* GP */
        results[0] = gp_simulate((void*)best_gp, &plant, steps, dt, &ws, tree_eval_wrap);

        /* PID: find best PID for this world (36 candidates) */
        {
            double best_sc = 1e100;
            int best_idx = -1;
            ControllerResult cand_r[36];
            int npid = 0;
            double kp_list[] = {0.8, 1.3, 2.0, 2.9};
            double ki_list[] = {0.0, 0.12, 0.28};
            double kd_list[] = {0.0, 0.08, 0.20};
            for (int ip = 0; ip < 4; ip++) {
                for (int ii = 0; ii < 3; ii++) {
                    for (int id = 0; id < 3; id++) {
                        ControllerResult r = pid_simulate(kp_list[ip], ki_list[ii], kd_list[id],
                                                          &plant, steps, dt, &ws);
                        double wcet, jitter;
                        controller_resource_metrics(profiles[1].cycles, profiles[1].ram,
                                                     profiles[1].branch, &wcet, &jitter);
                        double sc = controller_score(r.iae, r.overshoot, r.energy, wcet, jitter);
                        cand_r[npid] = r;
                        cand_r[npid].wcet_us = wcet;
                        cand_r[npid].jitter_us = jitter;
                        cand_r[npid].cycles = profiles[1].cycles;
                        cand_r[npid].ram_bytes = profiles[1].ram;
                        cand_r[npid].branch_points = profiles[1].branch;
                        cand_r[npid].score = sc;
                        if (sc < best_sc) { best_sc = sc; best_idx = npid; }
                        npid++;
                    }
                }
            }
            results[1] = cand_r[best_idx];
        }

        /* LQR */
        results[2] = lqr_simulate(K, &plant, steps, dt, &ws);

        /* MPC */
        results[3] = mpc_simulate(&plant, steps, dt, &ws);

        /* DSO = best PID (36 candidates, DSO resource profile) */
        {
            double best_sc = 1e100;
            int best_idx = -1;
            ControllerResult cand_r[36];
            int npid = 0;
            double kp_list[] = {0.8, 1.3, 2.0, 2.9};
            double ki_list[] = {0.0, 0.12, 0.28};
            double kd_list[] = {0.0, 0.08, 0.20};
            for (int ip = 0; ip < 4; ip++) {
                for (int ii = 0; ii < 3; ii++) {
                    for (int id = 0; id < 3; id++) {
                        ControllerResult r = pid_simulate(kp_list[ip], ki_list[ii], kd_list[id],
                                                          &plant, steps, dt, &ws);
                        double wcet, jitter;
                        controller_resource_metrics(profiles[4].cycles, profiles[4].ram,
                                                     profiles[4].branch, &wcet, &jitter);
                        double sc = controller_score(r.iae, r.overshoot, r.energy, wcet, jitter);
                        cand_r[npid] = r;
                        cand_r[npid].wcet_us = wcet;
                        cand_r[npid].jitter_us = jitter;
                        cand_r[npid].cycles = profiles[4].cycles;
                        cand_r[npid].ram_bytes = profiles[4].ram;
                        cand_r[npid].branch_points = profiles[4].branch;
                        cand_r[npid].score = sc;
                        if (sc < best_sc) { best_sc = sc; best_idx = npid; }
                        npid++;
                    }
                }
            }
            results[4] = cand_r[best_idx];
        }

        /* Set resource metrics + score for all controllers */
        for (int ci = 0; ci < N_CONTROLLERS; ci++) {
            ControllerResult *r = &results[ci];
            if (ci != 1 && ci != 4) { /* PID and DSO already have them */
                controller_resource_metrics(profiles[ci].cycles, profiles[ci].ram,
                                             profiles[ci].branch, &r->wcet_us, &r->jitter_us);
                r->cycles = profiles[ci].cycles;
                r->ram_bytes = profiles[ci].ram;
                r->branch_points = profiles[ci].branch;
                r->score = controller_score(r->iae, r->overshoot, r->energy,
                                             r->wcet_us, r->jitter_us);
            }
        }

        /* Accumulate */
        for (int ci = 0; ci < N_CONTROLLERS; ci++) {
            all_scores[ci][w] = results[ci].score;
            all_iae[ci][w]    = results[ci].iae;
            all_os[ci][w]     = results[ci].overshoot;
            all_en[ci][w]     = results[ci].energy;
            all_sat[ci][w]    = results[ci].saturated;
            all_cpass[ci][w]  = contract_pass(&c_default, &results[ci], steps,
                                               results[ci].cycles, results[ci].ram_bytes);
            sum_score[ci] += results[ci].score;
        }
    }

    /* ─── Output ───────────────────────────────────────────── */
    if (json_mode) {
        printf("{\n  \"n_worlds\": %d,\n  \"steps\": %d,\n  \"dt\": %.4f,\n", n_worlds, steps, dt);
        printf("  \"controllers\": [\n");
        for (int ci = 0; ci < N_CONTROLLERS; ci++) {
            double avg_sc = sum_score[ci] / n_worlds;
            double avg_iae = 0, avg_os = 0, avg_en = 0;
            double avg_sat = 0;
            int cp = 0;
            for (int w = 0; w < n_worlds; w++) {
                avg_iae += all_iae[ci][w];
                avg_os  += all_os[ci][w];
                avg_en  += all_en[ci][w];
                avg_sat += all_sat[ci][w];
                cp += all_cpass[ci][w];
            }
            avg_iae /= n_worlds; avg_os /= n_worlds;
            avg_en /= n_worlds; avg_sat /= n_worlds;
            printf("    {\n");
            printf("      \"name\": \"%s\",\n", profiles[ci].name);
            printf("      \"iae_mean\": %.6f,\n", avg_iae);
            printf("      \"overshoot_mean\": %.6f,\n", avg_os);
            printf("      \"energy_mean\": %.6f,\n", avg_en);
            printf("      \"saturation_frac\": %.6f,\n", avg_sat/steps);
            printf("      \"score_mean\": %.6f,\n", avg_sc);
            printf("      \"wcet_us\": %.4f,\n", all_iae[ci][0] == all_iae[ci][0] ? profiles[ci].cycles / 48.0 : 0);
            printf("      \"jitter_us\": %.4f,\n", 0.04 + 0.055 * profiles[ci].branch);
            printf("      \"cycles\": %d,\n", profiles[ci].cycles);
            printf("      \"ram_bytes\": %d,\n", profiles[ci].ram);
            printf("      \"branch_points\": %d,\n", profiles[ci].branch);
            printf("      \"contract_pass_rate\": %.4f%s\n",
                   (double)cp / n_worlds, ci < N_CONTROLLERS-1 ? "," : "");
            printf("    }%s\n", ci < N_CONTROLLERS-1 ? "," : "");
        }
        printf("  ],\n");
        /* Sign tests */
        printf("  \"sign_tests_vs_dso\": {\n");
        for (int ci = 0; ci < N_CONTROLLERS; ci++) {
            if (strcmp(profiles[ci].name, "DSO") == 0) continue;
            double pv = sign_test_pvalue(all_scores[ci], all_scores[4], n_worlds);
            printf("    \"%s\": %.6f%s\n", profiles[ci].name, pv,
                   ci < N_CONTROLLERS-1 ? "," : "");
        }
        printf("  }\n}\n");
    } else {
        printf("\n========== 5-WAY BENCHMARK: %d worlds ==========\n", n_worlds);
        printf("%-6s", "Ctrl");
        printf(" %10s %10s %10s %7s %10s %7s %7s %5s %4s %6s",
               "IAE", "Overshoot", "Energy", "Sat%", "Score",
               "WCETus", "Jitus", "Cyc", "RAM", "Ctr%");
        printf("\n");
        printf("------");
        printf(" ---------- ---------- ---------- ------- ---------- ------- ------- ----- ---- ------\n");

        for (int ci = 0; ci < N_CONTROLLERS; ci++) {
            double avg_sc = sum_score[ci] / n_worlds;
            double avg_iae = 0, avg_os = 0, avg_en = 0;
            double avg_sat = 0;
            int cp = 0;
            for (int w = 0; w < n_worlds; w++) {
                avg_iae += all_iae[ci][w];
                avg_os  += all_os[ci][w];
                avg_en  += all_en[ci][w];
                avg_sat += all_sat[ci][w];
                cp += all_cpass[ci][w];
            }
            avg_iae /= n_worlds; avg_os /= n_worlds;
            avg_en /= n_worlds; avg_sat /= n_worlds;

            printf("%-6s", profiles[ci].name);
            printf(" %10.6f %10.6f %10.6f %6.2f%% %10.6f",
                   avg_iae, avg_os, avg_en,
                   100.0 * avg_sat / steps,
                   avg_sc);
            printf(" %6.2f %6.3f %4d %4d %5.1f%%",
                   profiles[ci].cycles / 48.0,
                   0.04 + 0.055 * profiles[ci].branch,
                   profiles[ci].cycles,
                   profiles[ci].ram,
                   100.0 * (double)cp / n_worlds);
            printf("\n");
        }
        printf("------");
        printf(" ---------- ---------- ---------- ------- ---------- ------- ------- ----- ---- ------\n");

        /* Sign tests vs DSO */
        printf("\nSign tests (two-sided, vs DSO):\n");
        for (int ci = 0; ci < N_CONTROLLERS; ci++) {
            if (strcmp(profiles[ci].name, "DSO") == 0) continue;
            double pv = sign_test_pvalue(all_scores[ci], all_scores[4], n_worlds);
            int wins = 0, losses = 0;
            for (int w = 0; w < n_worlds; w++) {
                if (all_scores[ci][w] < all_scores[4][w]) wins++;
                else if (all_scores[ci][w] > all_scores[4][w]) losses++;
            }
            printf("  %s vs DSO: wins=%d losses=%d p=%.6f\n",
                   profiles[ci].name, wins, losses, pv);
        }

        /* Compare GP best vs each other */
        printf("\nGP improvement over others:\n");
        for (int ci = 1; ci < N_CONTROLLERS; ci++) {
            double gp_avg = sum_score[0] / n_worlds;
            double other_avg = sum_score[ci] / n_worlds;
            double impr = (other_avg - gp_avg) / (other_avg > 0 ? other_avg : 1) * 100;
            printf("  GP vs %s: %+.2f%%  (GP=%.4f, %s=%.4f)\n",
                   profiles[ci].name, impr, gp_avg, profiles[ci].name, other_avg);
        }
    }

    for (int ci = 0; ci < N_CONTROLLERS; ci++) {
        free(all_scores[ci]);
        free(all_iae[ci]);
        free(all_os[ci]);
        free(all_en[ci]);
        free(all_sat[ci]);
        free(all_cpass[ci]);
    }
}

/* ─── Multi-seed sweep ────────────────────────────────────────── */

static void run_sweep(int start_seed, int n_seeds, int n_worlds,
                       int steps, double dt, int pop_size, int generations,
                       int max_depth, double mut_rate, double cross_rate,
                       int tournament_size, double bloat_penalty)
{
    /* CSV header */
    printf("seed,gen,worlds,steps,fitness,tree_size,best_score_path\n");

    for (int s = 0; s < n_seeds; s++) {
        uint64_t seed = (uint64_t)(start_seed + s);

        GpPopulation pop = {0};
        gp_evolve(&pop, pop_size, generations, max_depth,
                  mut_rate, cross_rate, tournament_size,
                  n_worlds, steps, dt, bloat_penalty, seed);

        /* Run benchmark with best tree */
        for (int w = 0; w < n_worlds; w++) {
            uint64_t ws = seed + (uint64_t)w * 100000ULL;
            Plant plant = plant_random(&ws);
            ControllerResult r = gp_simulate((void*)&pop.trees[0], &plant,
                                              steps, dt, &ws, tree_eval_wrap);
            (void)r;
        }

        char tree_str[2048];
        gp_tree_print(&pop.trees[0], tree_str, sizeof(tree_str));

        printf("%d,%d,%d,%d,%.6f,%d,\"%s\"\n",
               start_seed + s, generations, n_worlds, steps,
               pop.trees[0].fitness, pop.trees[0].size, tree_str);

        free(pop.trees);
    }
}

/* ═══════════════════════════════════════════════════════════════ */
/*  MAIN                                                          */
/* ═══════════════════════════════════════════════════════════════ */

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
    int sweep_start = 0, sweep_count = 0;
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
        else if (strcmp(argv[i], "--sweep") == 0) {
            if (i+2 < argc) { sweep_start = atoi(argv[++i]); sweep_count = atoi(argv[++i]); }
            else { fprintf(stderr, "--sweep needs START COUNT\n"); return 1; }
        }
        else if (strcmp(argv[i], "--json") == 0)     json_mode = 1;
        else if (strcmp(argv[i], "--bloat") == 0 && i+1 < argc) bloat_penalty = atof(argv[++i]);
        else { fprintf(stderr, "Unknown: %s\n", argv[i]); return 1; }
    }

    /* Multi-seed sweep mode */
    if (sweep_count > 0) {
        run_sweep(sweep_start, sweep_count, n_worlds, steps, dt,
                  pop_size, generations, max_depth, mut_rate, cross_rate,
                  tournament_size, bloat_penalty);
        return 0;
    }

    /* Run evolution */
    GpPopulation pop = {0};

    printf("DSO GP — Genetic Programming for Controller Synthesis\n");
    printf("====================================================\n");
    printf("  pop=%d gen=%d depth=%d mut=%.2f cross=%.2f tourn=%d bloat=%.3f\n",
           pop_size, generations, max_depth, mut_rate, cross_rate, tournament_size, bloat_penalty);
    printf("  worlds=%d steps=%d dt=%.4f seed=%llu\n",
           n_worlds, steps, dt, (unsigned long long)seed);
    printf("\n");

    fflush(stdout);
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
        run_benchmark(benchmark, steps, dt, seed, &pop.trees[0], json_mode);
    }

    /* Cleanup */
    free(pop.trees);

    printf("\nDone.\n");
    return 0;
}
