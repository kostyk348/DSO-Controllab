#include "gp.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

/* ── splitmix64 RNG ──────────────────────────────────────────── */
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

/* ── Public RNG wrappers ─────────────────────────────────────── */
uint64_t gp_rand_u64(uint64_t *s)    { return splitmix64(s); }
double   gp_rand_uniform(uint64_t *s, double lo, double hi) { return runif(s, lo, hi); }
int      gp_rand_int(uint64_t *s, int lo, int hi) { return rint_range(s, lo, hi); }
double   gp_rand_normal(uint64_t *s, double mean, double std) {
    double u = runif(s, 0, 1), v = runif(s, 0, 1);
    if (u < 1e-15) u = 1e-15;
    return mean + std * sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
}

/* ── Subtree helpers ─────────────────────────────────────────── */
static int count_subtree(const GpNode *nodes, int idx) {
    if (idx < 0) return 0;
    int s = 1;
    if (nodes[idx].left  >= 0) s += count_subtree(nodes, nodes[idx].left);
    if (nodes[idx].right >= 0) s += count_subtree(nodes, nodes[idx].right);
    return s;
}

/* Copy subtree from src at src_idx to dst starting at dst_start.
 * Returns number of nodes written.  Writes at most GP_MAX_NODES. */
static int copy_subtree(GpNode *dst, int dst_start,
                        const GpNode *src, int src_idx) {
    if (src_idx < 0) return 0;
    if (dst_start < 0 || dst_start >= GP_MAX_NODES) return 0;
    dst[dst_start] = src[src_idx];
    int offset = 1;
    if (src[src_idx].left >= 0) {
        dst[dst_start].left = (int8_t)(dst_start + offset);
        offset += copy_subtree(dst, dst_start + offset, src, src[src_idx].left);
    } else {
        dst[dst_start].left = -1;
    }
    if (src[src_idx].right >= 0) {
        dst[dst_start].right = (int8_t)(dst_start + offset);
        offset += copy_subtree(dst, dst_start + offset, src, src[src_idx].right);
    } else {
        dst[dst_start].right = -1;
    }
    return offset;
}

/* ── gp_tree_random: grow a random tree ──────────────────────── */
void gp_tree_random(GpTree *t, int max_depth, uint64_t *rng) {
    NodeType funcs[] = {
        NODE_ADD, NODE_SUB, NODE_MUL, NODE_DIV,
        NODE_MIN, NODE_MAX, NODE_SQ, NODE_ABS, NODE_NEG
    };
    int n_funcs = 9;
    NodeType terms[] = {
        NODE_CONST, NODE_ERROR, NODE_INTEGRAL, NODE_DERIV, NODE_Y
    };
    int n_terms = 5;

    /* Recursive grow via explicit stack: store (pos, depth) pairs */
    int stack_pos[GP_MAX_NODES];
    int stack_depth[GP_MAX_NODES];
    int sp = 0;
    int next = 1; /* next free slot */

    stack_pos[sp] = 0;
    stack_depth[sp] = 0;
    sp++;

    while (sp > 0 && next < GP_MAX_NODES) {
        sp--;
        int pos = stack_pos[sp];
        int d   = stack_depth[sp];

        int is_terminal = (d >= max_depth) ? 1 : (runif(rng, 0, 1) < 0.5);

        if (is_terminal) {
            NodeType ty = terms[rint_range(rng, 0, n_terms - 1)];
            t->nodes[pos].type  = ty;
            t->nodes[pos].left  = -1;
            t->nodes[pos].right = -1;
            t->nodes[pos].value = (ty == NODE_CONST) ? runif(rng, -2.0, 2.0) : 0.0;
        } else {
            NodeType ty = funcs[rint_range(rng, 0, n_funcs - 1)];
            int is_unary = (ty == NODE_SQ || ty == NODE_ABS || ty == NODE_NEG);
        if (next >= GP_MAX_NODES - 2) {
            /* Not enough room — force terminal */
            if (next >= GP_MAX_NODES) next = GP_MAX_NODES - 1;
            NodeType fty = terms[rint_range(rng, 0, n_terms - 1)];
            t->nodes[pos].type  = fty;
            t->nodes[pos].left  = -1;
            t->nodes[pos].right = -1;
            t->nodes[pos].value = (fty == NODE_CONST) ? runif(rng, -2.0, 2.0) : 0.0;
            continue;
        }
        int left_pos = next++;
        int right_pos = is_unary ? -1 : (next < GP_MAX_NODES ? next++ : -1);

        if (right_pos < 0 && !is_unary) {
                /* Not enough room — make it unary or terminal instead */
                if (runif(rng, 0, 1) < 0.5 && next < GP_MAX_NODES) {
                    ty = NODE_ABS;
                    right_pos = -1;
                } else {
                    NodeType fty = terms[rint_range(rng, 0, n_terms - 1)];
                    t->nodes[pos].type  = fty;
                    t->nodes[pos].left  = -1;
                    t->nodes[pos].right = -1;
                    t->nodes[pos].value = (fty == NODE_CONST) ? runif(rng, -2.0, 2.0) : 0.0;
                    continue;
                }
            }

            t->nodes[pos].type  = ty;
            t->nodes[pos].left  = (int8_t)left_pos;
            t->nodes[pos].right = (int8_t)right_pos;
            t->nodes[pos].value = 0.0;

            if (right_pos >= 0) {
                stack_pos[sp] = right_pos; stack_depth[sp] = d + 1; sp++;
            }
            stack_pos[sp] = left_pos; stack_depth[sp] = d + 1; sp++;
        }
    }

    t->size = next;
    t->fitness = 0.0;
}

/* ── gp_tree_copy ────────────────────────────────────────────── */
void gp_tree_copy(GpTree *dst, const GpTree *src) {
    memcpy(dst, src, sizeof(GpTree));
}

/* ── gp_tree_eval: evaluate tree expression ──────────────────── */
double gp_tree_eval(const GpTree *t, double error, double integral,
                    double deriv, double y) {
    if (!t) return 0.0;
    int sz = t->size;
    if (sz <= 0 || sz > GP_MAX_NODES) return 0.0;

    /* Post-order iterative evaluation */
    int stack[GP_MAX_NODES];
    double val[GP_MAX_NODES];
    int sp = 0;
    char visited[GP_MAX_NODES];
    memset(visited, 0, GP_MAX_NODES);

    /* Validate root */
    if (t->nodes[0].type < NODE_ADD || t->nodes[0].type > NODE_Y) return 0.0;

    stack[sp++] = 0;
    int node_count = 0;

    while (sp > 0 && node_count < GP_MAX_NODES * 2) {
        node_count++;
        int idx = stack[sp - 1];

        /* Bounds check */
        if (idx < 0 || idx >= sz) { sp--; continue; }

        if (visited[idx]) {
            sp--;
            const GpNode *n = &t->nodes[idx];

            /* Validate left/right indices */
            int l = n->left, r = n->right;

            switch (n->type) {
                case NODE_CONST:    val[idx] = n->value; break;
                case NODE_ERROR:    val[idx] = error; break;
                case NODE_INTEGRAL: val[idx] = integral; break;
                case NODE_DERIV:    val[idx] = deriv; break;
                case NODE_Y:        val[idx] = y; break;
                case NODE_ADD:
                    val[idx] = (l >= 0 && l < sz ? val[l] : 0)
                             + (r >= 0 && r < sz ? val[r] : 0);
                    break;
                case NODE_SUB:
                    val[idx] = (l >= 0 && l < sz ? val[l] : 0)
                             - (r >= 0 && r < sz ? val[r] : 0);
                    break;
                case NODE_MUL:
                    val[idx] = (l >= 0 && l < sz ? val[l] : 0)
                             * (r >= 0 && r < sz ? val[r] : 0);
                    break;
                case NODE_DIV: {
                    double denom = (r >= 0 && r < sz) ? val[r] : 1.0;
                    val[idx] = (fabs(denom) < 1e-10) ? 1.0
                             : ((l >= 0 && l < sz ? val[l] : 0) / denom);
                    break;
                }
                case NODE_MIN:
                    val[idx] = fmin(l >= 0 && l < sz ? val[l] : 0,
                                    r >= 0 && r < sz ? val[r] : 0);
                    break;
                case NODE_MAX:
                    val[idx] = fmax(l >= 0 && l < sz ? val[l] : 0,
                                    r >= 0 && r < sz ? val[r] : 0);
                    break;
                case NODE_SQ:
                    val[idx] = (l >= 0 && l < sz ? val[l] : 0)
                             * (l >= 0 && l < sz ? val[l] : 0);
                    break;
                case NODE_ABS:
                    val[idx] = fabs(l >= 0 && l < sz ? val[l] : 0);
                    break;
                case NODE_NEG:
                    val[idx] = -(l >= 0 && l < sz ? val[l] : 0);
                    break;
                default:
                    val[idx] = 0.0;
                    break;
            }
            if (val[idx] >  1e6) val[idx] =  1e6;
            if (val[idx] < -1e6) val[idx] = -1e6;
        } else {
            visited[idx] = 1;
            if (t->nodes[idx].right >= 0 && t->nodes[idx].right < sz &&
                sp < GP_MAX_NODES)
                stack[sp++] = t->nodes[idx].right;
            if (t->nodes[idx].left >= 0 && t->nodes[idx].left < sz &&
                sp < GP_MAX_NODES)
                stack[sp++] = t->nodes[idx].left;
        }
    }

    if (sz <= 0) return 0.0;
    double result = (sz > 0 && !isnan(val[0]) && !isinf(val[0])) ? val[0] : 0.0;
    if (result >  4.0) result =  4.0;
    if (result < -4.0) result = -4.0;
    return result;
}

/* ── Pick a random node (bias toward internal) ───────────────── */
static int pick_node(const GpTree *t, uint64_t *rng) {
    if (t->size <= 1) return 0;
    for (int attempt = 0; attempt < 10; attempt++) {
        int idx = rint_range(rng, 0, t->size - 1);
        NodeType ty = t->nodes[idx].type;
        int is_term = (ty == NODE_CONST || ty == NODE_ERROR ||
                       ty == NODE_INTEGRAL || ty == NODE_DERIV || ty == NODE_Y);
        if (!is_term) return idx;
    }
    return rint_range(rng, 0, t->size - 1);
}

/* ── gp_tree_mutate ──────────────────────────────────────────── */
void gp_tree_mutate(GpTree *t, int max_depth, uint64_t *rng) {
    if (t->size <= 0) return;
    int idx = rint_range(rng, 0, t->size - 1);
    GpNode *n = &t->nodes[idx];
    NodeType terms[] = {NODE_CONST, NODE_ERROR, NODE_INTEGRAL, NODE_DERIV, NODE_Y};
    int n_terms = 5;

    int is_term = (n->type == NODE_CONST || n->type == NODE_ERROR ||
                   n->type == NODE_INTEGRAL || n->type == NODE_DERIV || n->type == NODE_Y);

    if (is_term) {
        if (n->type == NODE_CONST && runif(rng, 0, 1) < 0.5) {
            n->value += gp_rand_normal(rng, 0.0, 0.3);
        } else {
            NodeType nt = terms[rint_range(rng, 0, n_terms - 1)];
            n->type  = nt;
            n->left  = -1;
            n->right = -1;
            n->value = (nt == NODE_CONST) ? runif(rng, -2.0, 2.0) : 0.0;
        }
    } else {
        /* Internal node: replace subtree with a tiny random one */
        int sub_sz = count_subtree(t->nodes, idx);
        int free_slots = GP_MAX_NODES - (t->size - sub_sz);
        if (free_slots < 2) {
            /* Just change the op type */
            NodeType funcs[] = {NODE_ADD,NODE_SUB,NODE_MUL,NODE_DIV,
                                NODE_MIN,NODE_MAX,NODE_SQ,NODE_ABS,NODE_NEG};
            n->type = funcs[rint_range(rng, 0, 8)];
            return;
        }

        GpTree tmp;
        memset(&tmp, 0, sizeof(tmp));
        gp_tree_random(&tmp, max_depth > 1 ? 1 : 0, rng);
        int new_sz = count_subtree(tmp.nodes, 0);
        if (new_sz <= free_slots && new_sz > 0) {
            copy_subtree(t->nodes, idx, tmp.nodes, 0);
            t->size = count_subtree(t->nodes, 0);
        }
    }
}

/* ── gp_tree_crossover ───────────────────────────────────────── */
void gp_tree_crossover(GpTree *a, GpTree *b, uint64_t *rng) {
    if (a->size < 2 || b->size < 2) return;

    int i = pick_node(a, rng);
    int j = pick_node(b, rng);

    int sz_i = count_subtree(a->nodes, i);
    int sz_j = count_subtree(b->nodes, j);

    int a_free = GP_MAX_NODES - (a->size - sz_i);
    int b_free = GP_MAX_NODES - (b->size - sz_j);

    if (sz_j > a_free || sz_i > b_free) return;

    /* Save copies */
    GpNode tmp_a[GP_MAX_NODES];
    GpNode tmp_b[GP_MAX_NODES];
    copy_subtree(tmp_a, 0, a->nodes, i);
    copy_subtree(tmp_b, 0, b->nodes, j);

    /* Swap */
    copy_subtree(a->nodes, i, tmp_b, 0);
    copy_subtree(b->nodes, j, tmp_a, 0);

    a->size = count_subtree(a->nodes, 0);
    b->size = count_subtree(b->nodes, 0);
}

/* ── gp_tree_print ───────────────────────────────────────────── */
static void print_rec(const GpTree *t, int idx, char *buf, int cap) {
    if (idx < 0 || idx >= t->size) { strncat(buf, "?", cap - strlen(buf) - 1); return; }
    const GpNode *n = &t->nodes[idx];
    char tmp[64];

    switch (n->type) {
        case NODE_CONST:
            snprintf(tmp, sizeof(tmp), "%.4f", n->value);
            strncat(buf, tmp, cap - strlen(buf) - 1);
            break;
        case NODE_ERROR:    strncat(buf, "ERROR", cap - strlen(buf) - 1); break;
        case NODE_INTEGRAL: strncat(buf, "INT", cap - strlen(buf) - 1); break;
        case NODE_DERIV:    strncat(buf, "DER", cap - strlen(buf) - 1); break;
        case NODE_Y:        strncat(buf, "Y", cap - strlen(buf) - 1); break;
        case NODE_ADD: case NODE_SUB: case NODE_MUL: case NODE_DIV: {
            char op = (n->type == NODE_ADD) ? '+' : (n->type == NODE_SUB) ? '-' :
                      (n->type == NODE_MUL) ? '*' : '/';
            snprintf(tmp, sizeof(tmp), "(%c ", op);
            strncat(buf, tmp, cap - strlen(buf) - 1);
            print_rec(t, n->left, buf, cap);
            strncat(buf, " ", cap - strlen(buf) - 1);
            print_rec(t, n->right, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
        }
        case NODE_MIN:
            strncat(buf, "(min ", cap - strlen(buf) - 1);
            print_rec(t, n->left, buf, cap);
            strncat(buf, " ", cap - strlen(buf) - 1);
            print_rec(t, n->right, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
        case NODE_MAX:
            strncat(buf, "(max ", cap - strlen(buf) - 1);
            print_rec(t, n->left, buf, cap);
            strncat(buf, " ", cap - strlen(buf) - 1);
            print_rec(t, n->right, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
        case NODE_SQ:
            strncat(buf, "(sq ", cap - strlen(buf) - 1);
            print_rec(t, n->left, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
        case NODE_ABS:
            strncat(buf, "(abs ", cap - strlen(buf) - 1);
            print_rec(t, n->left, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
        case NODE_NEG:
            strncat(buf, "(- ", cap - strlen(buf) - 1);
            print_rec(t, n->left, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
    }
}

void gp_tree_print(const GpTree *t, char *buf, int buf_size) {
    if (!buf || buf_size <= 0) return;
    buf[0] = '\0';
    if (t->size <= 0) { snprintf(buf, buf_size, "EMPTY"); return; }
    print_rec(t, 0, buf, buf_size);
}
