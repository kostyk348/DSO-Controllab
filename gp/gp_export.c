#include "gp.h"
#include <stdio.h>
#include <string.h>

/* ── Write expression for C ──────────────────────────────────── */
static void write_expr_c(const GpTree *t, int idx, char *buf, int cap) {
    if (idx < 0 || idx >= t->size) { strncat(buf, "0", cap - strlen(buf) - 1); return; }
    const GpNode *n = &t->nodes[idx];
    char tmp[128];

    switch (n->type) {
        case NODE_CONST:
            snprintf(tmp, sizeof(tmp), "%.6f", n->value);
            strncat(buf, tmp, cap - strlen(buf) - 1);
            break;
        case NODE_ERROR:    strncat(buf, "error", cap - strlen(buf) - 1); break;
        case NODE_INTEGRAL: strncat(buf, "integral", cap - strlen(buf) - 1); break;
        case NODE_DERIV:    strncat(buf, "deriv", cap - strlen(buf) - 1); break;
        case NODE_Y:        strncat(buf, "y", cap - strlen(buf) - 1); break;
        case NODE_ADD:
            strncat(buf, "(", cap - strlen(buf) - 1);
            write_expr_c(t, n->left, buf, cap);
            strncat(buf, " + ", cap - strlen(buf) - 1);
            write_expr_c(t, n->right, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
        case NODE_SUB:
            strncat(buf, "(", cap - strlen(buf) - 1);
            write_expr_c(t, n->left, buf, cap);
            strncat(buf, " - ", cap - strlen(buf) - 1);
            write_expr_c(t, n->right, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
        case NODE_MUL:
            strncat(buf, "(", cap - strlen(buf) - 1);
            write_expr_c(t, n->left, buf, cap);
            strncat(buf, " * ", cap - strlen(buf) - 1);
            write_expr_c(t, n->right, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
        case NODE_DIV:
            strncat(buf, "(", cap - strlen(buf) - 1);
            write_expr_c(t, n->left, buf, cap);
            strncat(buf, " / ", cap - strlen(buf) - 1);
            write_expr_c(t, n->right, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
        case NODE_MIN:
            strncat(buf, "fmin(", cap - strlen(buf) - 1);
            write_expr_c(t, n->left, buf, cap);
            strncat(buf, ", ", cap - strlen(buf) - 1);
            write_expr_c(t, n->right, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
        case NODE_MAX:
            strncat(buf, "fmax(", cap - strlen(buf) - 1);
            write_expr_c(t, n->left, buf, cap);
            strncat(buf, ", ", cap - strlen(buf) - 1);
            write_expr_c(t, n->right, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
        case NODE_SQ:
            strncat(buf, "((", cap - strlen(buf) - 1);
            write_expr_c(t, n->left, buf, cap);
            strncat(buf, ") * (", cap - strlen(buf) - 1);
            write_expr_c(t, n->left, buf, cap);
            strncat(buf, "))", cap - strlen(buf) - 1);
            break;
        case NODE_ABS:
            strncat(buf, "fabs(", cap - strlen(buf) - 1);
            write_expr_c(t, n->left, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
        case NODE_NEG:
            strncat(buf, "(-(", cap - strlen(buf) - 1);
            write_expr_c(t, n->left, buf, cap);
            strncat(buf, "))", cap - strlen(buf) - 1);
            break;
        case NODE_SIN:
            strncat(buf, "sin(", cap - strlen(buf) - 1);
            write_expr_c(t, n->left, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
        case NODE_COS:
            strncat(buf, "cos(", cap - strlen(buf) - 1);
            write_expr_c(t, n->left, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
    }
}

/* ── Write expression for Ada SPARK ──────────────────────────── */
static void write_expr_ada(const GpTree *t, int idx, char *buf, int cap) {
    if (idx < 0 || idx >= t->size) { strncat(buf, "0.0", cap - strlen(buf) - 1); return; }
    const GpNode *n = &t->nodes[idx];
    char tmp[128];

    switch (n->type) {
        case NODE_CONST:
            snprintf(tmp, sizeof(tmp), "%.6f", n->value);
            strncat(buf, tmp, cap - strlen(buf) - 1);
            break;
        case NODE_ERROR:    strncat(buf, "Error", cap - strlen(buf) - 1); break;
        case NODE_INTEGRAL: strncat(buf, "Integral", cap - strlen(buf) - 1); break;
        case NODE_DERIV:    strncat(buf, "Deriv", cap - strlen(buf) - 1); break;
        case NODE_Y:        strncat(buf, "Y", cap - strlen(buf) - 1); break;
        case NODE_ADD:
            strncat(buf, "(", cap - strlen(buf) - 1);
            write_expr_ada(t, n->left, buf, cap);
            strncat(buf, " + ", cap - strlen(buf) - 1);
            write_expr_ada(t, n->right, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
        case NODE_SUB:
            strncat(buf, "(", cap - strlen(buf) - 1);
            write_expr_ada(t, n->left, buf, cap);
            strncat(buf, " - ", cap - strlen(buf) - 1);
            write_expr_ada(t, n->right, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
        case NODE_MUL:
            strncat(buf, "(", cap - strlen(buf) - 1);
            write_expr_ada(t, n->left, buf, cap);
            strncat(buf, " * ", cap - strlen(buf) - 1);
            write_expr_ada(t, n->right, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
        case NODE_DIV:
            strncat(buf, "Safe_Div(", cap - strlen(buf) - 1);
            write_expr_ada(t, n->left, buf, cap);
            strncat(buf, ", ", cap - strlen(buf) - 1);
            write_expr_ada(t, n->right, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
        case NODE_MIN:
            strncat(buf, "Float'Min(", cap - strlen(buf) - 1);
            write_expr_ada(t, n->left, buf, cap);
            strncat(buf, ", ", cap - strlen(buf) - 1);
            write_expr_ada(t, n->right, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
        case NODE_MAX:
            strncat(buf, "Float'Max(", cap - strlen(buf) - 1);
            write_expr_ada(t, n->left, buf, cap);
            strncat(buf, ", ", cap - strlen(buf) - 1);
            write_expr_ada(t, n->right, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
        case NODE_SQ:
            strncat(buf, "((", cap - strlen(buf) - 1);
            write_expr_ada(t, n->left, buf, cap);
            strncat(buf, ") * (", cap - strlen(buf) - 1);
            write_expr_ada(t, n->left, buf, cap);
            strncat(buf, "))", cap - strlen(buf) - 1);
            break;
        case NODE_ABS:
            strncat(buf, "abs (", cap - strlen(buf) - 1);
            write_expr_ada(t, n->left, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
        case NODE_NEG:
            strncat(buf, "(-(", cap - strlen(buf) - 1);
            write_expr_ada(t, n->left, buf, cap);
            strncat(buf, "))", cap - strlen(buf) - 1);
            break;
        case NODE_SIN:
            strncat(buf, "Sin (", cap - strlen(buf) - 1);
            write_expr_ada(t, n->left, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
        case NODE_COS:
            strncat(buf, "Cos (", cap - strlen(buf) - 1);
            write_expr_ada(t, n->left, buf, cap);
            strncat(buf, ")", cap - strlen(buf) - 1);
            break;
    }
}

/* ── Export as C function ────────────────────────────────────── */

void gp_export_c(const GpTree *t, const char *func_name, FILE *out) {
    char expr[4096];
    expr[0] = '\0';
    write_expr_c(t, 0, expr, sizeof(expr));

    fprintf(out, "/* Auto-generated by DSO GP — fitness=%.6f */\n", t->fitness);
    fprintf(out, "#include <math.h>\n\n");
    fprintf(out, "static inline double clamp(double x, double lo, double hi) {\n");
    fprintf(out, "    return x < lo ? lo : (x > hi ? hi : x);\n");
    fprintf(out, "}\n\n");
    fprintf(out, "double %s(double error, double integral, double deriv, double y) {\n", func_name);
    fprintf(out, "    double u = %s;\n", expr);
    fprintf(out, "    if (isnan(u) || isinf(u)) return 0.0;\n");
    fprintf(out, "    return clamp(u, -4.0, 4.0);\n");
    fprintf(out, "}\n");
}

/* ── Export as Ada SPARK function ────────────────────────────── */

void gp_export_ada(const GpTree *t, const char *func_name, FILE *out) {
    /* Write spec (.ads) */
    fprintf(out, "-- Auto-generated by DSO GP — fitness=%.6f\n", t->fitness);
    fprintf(out, "package Controller with SPARK_Mode is\n\n");
    fprintf(out, "   function Compute\n");
    fprintf(out, "     (Error, Integral, Deriv, Y : in Float) return Float\n");
    fprintf(out, "     with Pre  => Error   in -10.0 .. 10.0 and\n");
    fprintf(out, "                  Integral in -10.0 .. 10.0 and\n");
    fprintf(out, "                  Deriv   in -10.0 .. 10.0 and\n");
    fprintf(out, "                  Y       in -10.0 .. 10.0,\n");
    fprintf(out, "          Post => Compute'Result in -4.0 .. 4.0;\n\n");
    fprintf(out, "end Controller;\n");
    fprintf(out, "\n");
}

void gp_export_ada_body(const GpTree *t, const char *func_name, FILE *out) {
    char expr[4096];
    expr[0] = '\0';
    write_expr_ada(t, 0, expr, sizeof(expr));

    fprintf(out, "-- Auto-generated by DSO GP — fitness=%.6f\n", t->fitness);
    fprintf(out, "with Ada.Numerics.Elementary_Functions;\n");
    fprintf(out, "use Ada.Numerics.Elementary_Functions;\n");
    fprintf(out, "package body Controller with SPARK_Mode is\n\n");
    fprintf(out, "   function Safe_Div (X, Y : Float) return Float is\n");
    fprintf(out, "   begin\n");
    fprintf(out, "      if abs Y < 1.0e-10 then\n");
    fprintf(out, "         return 1.0;\n");
    fprintf(out, "      else\n");
    fprintf(out, "         return X / Y;\n");
    fprintf(out, "      end if;\n");
    fprintf(out, "   end Safe_Div;\n\n");
    fprintf(out, "   function Compute\n");
    fprintf(out, "     (Error, Integral, Deriv, Y : in Float) return Float\n");
    fprintf(out, "   is\n");
    fprintf(out, "      Raw : Float;\n");
    fprintf(out, "      U   : Float;\n");
    fprintf(out, "   begin\n");
    fprintf(out, "      Raw := %s;\n", expr);
    fprintf(out, "      if Raw < -4.0 then\n");
    fprintf(out, "         U := -4.0;\n");
    fprintf(out, "      elsif Raw > 4.0 then\n");
    fprintf(out, "         U := 4.0;\n");
    fprintf(out, "      else\n");
    fprintf(out, "         U := Raw;\n");
    fprintf(out, "      end if;\n");
    fprintf(out, "      return U;\n");
    fprintf(out, "   end Compute;\n\n");
    fprintf(out, "end Controller;\n");
}
