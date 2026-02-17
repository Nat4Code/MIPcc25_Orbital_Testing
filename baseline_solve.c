/*
 * baseline_solve.c
 *
 * Plain Gurobi C API solve baseline, but emits a FINAL JSON line for benchmarking scripts.
 *
 * Usage:
 *   ./baseline model.mps [time_limit_seconds]
 *
 * Notes:
 * - We intentionally print the JSON line LAST so your Python extract_last_json() grabs it,
 *   even if Gurobi logs a lot of text before that.
 * - We keep OutputFlag=1 by default (as you had it). If you want fully silent, set to 0.
 */

#include "gurobi_c.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---------- timing helpers (wall-clock) ---------- */
static double wall_now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ---------- JSON helpers (minimal escaping) ---------- */
static void json_print_escaped_string(const char *s) {
  putchar('"');
  if (s) {
    for (const unsigned char *p = (const unsigned char*)s; *p; ++p) {
      unsigned char c = *p;
      switch (c) {
        case '\\': fputs("\\\\", stdout); break;
        case '"':  fputs("\\\"", stdout); break;
        case '\n': fputs("\\n", stdout); break;
        case '\r': fputs("\\r", stdout); break;
        case '\t': fputs("\\t", stdout); break;
        default:
          if (c < 0x20) {
            /* control chars -> \u00XX */
            printf("\\u%04x", (unsigned)c);
          } else {
            putchar((int)c);
          }
      }
    }
  }
  putchar('"');
}

static const char* status_name(int st) {
  switch (st) {
    case GRB_LOADED: return "LOADED";
    case GRB_OPTIMAL: return "OPTIMAL";
    case GRB_INFEASIBLE: return "INFEASIBLE";
    case GRB_INF_OR_UNBD: return "INF_OR_UNBD";
    case GRB_UNBOUNDED: return "UNBOUNDED";
    case GRB_CUTOFF: return "CUTOFF";
    case GRB_ITERATION_LIMIT: return "ITERATION_LIMIT";
    case GRB_NODE_LIMIT: return "NODE_LIMIT";
    case GRB_TIME_LIMIT: return "TIME_LIMIT";
    case GRB_SOLUTION_LIMIT: return "SOLUTION_LIMIT";
    case GRB_INTERRUPTED: return "INTERRUPTED";
    case GRB_NUMERIC: return "NUMERIC";
    case GRB_SUBOPTIMAL: return "SUBOPTIMAL";
    case GRB_INPROGRESS: return "INPROGRESS";
    default: return "STATUS_UNKNOWN";
  }
}

int main(int argc, char *argv[])
{
  int error = 0;
  GRBenv   *env   = NULL;
  GRBmodel *model = NULL;

  const char *model_file = NULL;
  double time_limit = -1.0;  /* < 0 means: no limit set */

  /* benchmark JSON fields */
  int ok = 0;
  int status = -1;
  int terminated_early = 0;
  double best_obj = 0.0;
  int have_obj = 0;
  double wall_runtime = 0.0;
  double grb_runtime = -1.0;
  const char *err_msg = NULL;
  
  // sets wall clock start time
  double t0 = wall_now_sec();

  if (argc < 2) {
    err_msg = "Usage: baseline model.mps [time_limit_seconds]";
    goto QUIT;
  }

  model_file = argv[1];
  if (argc >= 3) {
    time_limit = atof(argv[2]);
  }

  /* create environment and log file */
  error = GRBloadenv(&env, "baseline_solve.log");
  if (error) {
    err_msg = "GRBloadenv failed";
    goto QUIT;
  }

  /* keep output visible (set to 0 to silence) */
  error = GRBsetintparam(env, GRB_INT_PAR_OUTPUTFLAG, 1);
  if (error) {
    err_msg = "GRBsetintparam(OutputFlag) failed";
    goto QUIT;
  }

  /* Read model from file */
  error = GRBreadmodel(env, model_file, &model);
  if (error) {
    err_msg = "GRBreadmodel failed";
    goto QUIT;
  }

  /* Apply time limit if provided */
  if (time_limit > 0.0) {
    error = GRBsetdblparam(env, GRB_DBL_PAR_TIMELIMIT, time_limit);
    if (error) {
      err_msg = "GRBsetdblparam(TimeLimit) failed";
      goto QUIT;
    }
  }

  /* Optimize model */
  error = GRBoptimize(model);
  if (error) {
    err_msg = "GRBoptimize failed";
    goto QUIT;
  }

  /* Get solve status */
  error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &status);
  if (error) {
    err_msg = "GRBgetintattr(Status) failed";
    goto QUIT;
  }
  if (status == GRB_TIME_LIMIT) terminated_early = 1;

  /* Get objective if meaningful */
  if (status == GRB_OPTIMAL || status == GRB_TIME_LIMIT || status == GRB_SUBOPTIMAL) {
    double objval = 0.0;
    error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &objval);
    if (!error) {
      best_obj = objval;
      have_obj = 1;
    } else {
      /* If objective fetch fails, don't fail the whole run; just omit best_obj */
      error = 0;
    }
  }

  /* Gurobi-reported runtime (optional; nice to compare against wall time) */
  {
    double rt = 0.0;
    int e2 = GRBgetdblattr(model, GRB_DBL_ATTR_RUNTIME, &rt);
    if (!e2) grb_runtime = rt;
  }

  ok = 1;

QUIT:
  // finalize timing by grabbing end time and subtract from start:
  wall_runtime = wall_now_sec() - t0;

  /* Always print the JSON LAST so the Python script can parse it */
  {
    printf("{");
    printf("\"tool\":\"baseline\",");
    printf("\"ok\":%s,", ok ? "true" : "false");

    printf("\"mps\":");
    json_print_escaped_string(model_file ? model_file : "");
    printf(",");

    printf("\"time_limit_sec\":%.6f,", time_limit);

    printf("\"terminated_early\":%s,", terminated_early ? "true" : "false");

    printf("\"status_code\":%d,", status);
    printf("\"status\":");
    json_print_escaped_string(status_name(status));
    printf(",");

    /* baseline has no orbit info, but your scripts sometimes expect it */
    printf("\"orbit_count\":null,");

    if (have_obj) {
      printf("\"best_obj\":%.17g,", best_obj);
    } else {
      printf("\"best_obj\":null,");
    }

    printf("\"wall_runtime_sec\":%.9f,", wall_runtime);
    if (grb_runtime >= 0.0) {
      printf("\"grb_runtime_sec\":%.9f", grb_runtime);
    } else {
      printf("\"grb_runtime_sec\":null");
    }

    if (!ok) {
      printf(",\"error_msg\":");
      if (err_msg) json_print_escaped_string(err_msg);
      else if (env) json_print_escaped_string(GRBgeterrormsg(env));
      else json_print_escaped_string("unknown error");
    }

    printf("}\n");
    fflush(stdout);
  }

  if (model) GRBfreemodel(model);
  if (env)   GRBfreeenv(env);

  return ok ? 0 : 1;
}
