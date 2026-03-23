// heuristic_full.cpp
//
// Replica of heuristic_optimal.cpp pipeline + two-stage solve:
//   Stage 1: time-capped solve (default 4s) with MIP start.
//   Stage 2: hand off to exact solve (remove time limit) to proven optimality.
//
// New:
//   --verbose  : show live Gurobi log output in terminal
//
// Build:
//   g++ -O3 -std=c++20 heuristic_full.cpp -o heuristic_full \
//     -I/opt/gurobi1203/linux64/include -L/opt/gurobi1203/linux64/lib \
//     -lgurobi_c++ -lgurobi120
//
// Run:
//   ./heuristic_full model.mps --verbose
//   ./heuristic_full model.mps --stage1 4.0 --verbose
//

#include <gurobi_c++.h>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>

using namespace std;

static double now_sec() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static bool is_nan(double x) { return std::isnan(x); }

static bool has_flag(int argc, char** argv, const string& flag) {
  for (int i = 2; i < argc; ++i) if (string(argv[i]) == flag) return true;
  return false;
}

static double parse_flag_double(int argc, char** argv, const string& flag, double defval) {
  for (int i = 2; i + 1 < argc; ++i) {
    if (string(argv[i]) == flag) {
      try { return stod(argv[i+1]); } catch (...) { return defval; }
    }
  }
  return defval;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: heuristic_full model.mps [--stage1 4.0] [--verbose]\n";
    return 1;
  }

  const string mps = argv[1];
  const double STAGE1_LIMIT = parse_flag_double(argc, argv, "--stage1", 3000.0);
  const bool VERBOSE = has_flag(argc, argv, "--verbose");

  double t0 = now_sec();
  double t_graph=0, t_color=0, t_relax=0, t_score=0;

  try {
    GRBEnv env(true);

    // Live logs on demand
    env.set(GRB_IntParam_LogToConsole, VERBOSE ? 1 : 0);
    env.start();

    GRBModel model(env, mps);
    model.set(GRB_IntParam_OutputFlag, VERBOSE ? 1 : 0);

    const int numVars = model.get(GRB_IntAttr_NumVars);

    // gather variable info (use indices throughout)
    GRBVar* vars = model.getVars();
    vector<string> var_name(numVars);
    vector<char>   var_vtype(numVars);
    vector<double> var_obj(numVars), var_lb(numVars), var_ub(numVars);
    unordered_map<string,int> idx_by_name;
    idx_by_name.reserve((size_t)numVars * 2);

    for (int i = 0; i < numVars; ++i) {
      string nm = vars[i].get(GRB_StringAttr_VarName);
      var_name[i] = nm;
      idx_by_name[nm] = i;
      var_vtype[i] = vars[i].get(GRB_CharAttr_VType);
      var_obj[i]   = vars[i].get(GRB_DoubleAttr_Obj);
      var_lb[i]    = vars[i].get(GRB_DoubleAttr_LB);
      var_ub[i]    = vars[i].get(GRB_DoubleAttr_UB);
    }

    // build incidence graph (lightweight adjacency with abs coeffs)
    unordered_map<string, unordered_map<string,double>> G;

    const int numCons = model.get(GRB_IntAttr_NumConstrs);
    GRBConstr* cons = model.getConstrs();

    double t0_section = now_sec();
    for (int ci = 0; ci < numCons; ++ci) {
      string cname = cons[ci].get(GRB_StringAttr_ConstrName);
      GRBLinExpr row = model.getRow(cons[ci]);
      int rsz = row.size();
      for (int j = 0; j < rsz; ++j) {
        string vnm = row.getVar(j).get(GRB_StringAttr_VarName);
        double a = row.getCoeff(j);
        double w = fabs(a);
        G[vnm][cname] += w;
        G[cname][vnm] += w;
      }
    }
    t_graph += now_sec() - t0_section;

    const int numQ = model.get(GRB_IntAttr_NumQConstrs);
    GRBQConstr* qcons = model.getQConstrs();

    t0_section = now_sec();
    for (int qi = 0; qi < numQ; ++qi) {
      string qname = string("Q|") + to_string(qi);
      GRBQuadExpr qexpr = model.getQCRow(qcons[qi]);

      // linear part
      GRBLinExpr lin = qexpr.getLinExpr();
      int lsz = lin.size();
      for (int j = 0; j < lsz; ++j) {
        string vnm = lin.getVar(j).get(GRB_StringAttr_VarName);
        double a = lin.getCoeff(j);
        double w = fabs(a);
        G[vnm][qname] += w;
        G[qname][vnm] += w;
      }

      // quadratic part not expanded (replica of heuristic_optimal behavior)
      (void)qexpr;
    }
    t_graph += now_sec() - t0_section;

    // initial coloring
    unordered_map<string, vector<string>> color_classes;
    unordered_map<string,int> node_to_color;
    int color_id = 0;

    for (int ci = 0; ci < numCons; ++ci) {
      string cname = cons[ci].get(GRB_StringAttr_ConstrName);
      double rhs = cons[ci].get(GRB_DoubleAttr_RHS);
      char sense = cons[ci].get(GRB_CharAttr_Sense);
      string key = string("C|") + sense + "|" + to_string((long long)llround(rhs * 1e9));
      color_classes[key].push_back(cname);
    }
    for (int qi = 0; qi < numQ; ++qi) {
      string cname = string("q_cons") + to_string(qi);
      double rhs = qcons[qi].get(GRB_DoubleAttr_QCRHS);
      char sense = qcons[qi].get(GRB_CharAttr_QCSense);
      string key = string("Q|") + sense + "|" + to_string((long long)llround(rhs * 1e9));
      color_classes[key].push_back(cname);
    }
    for (int i = 0; i < numVars; ++i) {
      string key =
        string("V|") + var_vtype[i] +
        "|obj:" + to_string((long long)llround(var_obj[i] * 1e9)) +
        "|lb:"  + to_string((long long)llround(var_lb[i]  * 1e9)) +
        "|ub:"  + to_string((long long)llround(var_ub[i]  * 1e9));
      color_classes[key].push_back(var_name[i]);
    }
    for (auto &p : color_classes) {
      int cid = color_id++;
      for (auto &n : p.second) node_to_color[n] = cid;
    }

    // refine by weighted degree
    t0_section = now_sec();
    unordered_map<int, vector<string>> new_classes;
    for (auto &p : color_classes) {
      unordered_map<long long, vector<string>> tmp;
      for (auto &n : p.second) {
        long long deg = 0;
        auto itg = G.find(n);
        if (itg != G.end()) {
          for (auto &kv : itg->second) deg += (long long)llround(kv.second * 1000.0);
        }
        tmp[deg].push_back(n);
      }
      for (auto &q : tmp) {
        int cid = color_id++;
        for (auto &n : q.second) new_classes[cid].push_back(n);
      }
    }
    t_color += now_sec() - t0_section;

    unordered_map<int, vector<int>> var_partitions;
    for (auto &c : new_classes) {
      for (auto &n : c.second) {
        auto it = idx_by_name.find(n);
        if (it != idx_by_name.end()) var_partitions[c.first].push_back(it->second);
      }
    }

    // relaxation
    GRBModel relax = model.relax();
    relax.set(GRB_IntParam_OutputFlag, 0); // keep relaxation quiet even in verbose mode

    // folding constraints on relaxation
    for (auto &p : var_partitions) {
      auto &nodes = p.second;
      if (nodes.size() <= 1) continue;
      int first_vid = nodes[0];
      const string &first_nm = var_name[first_vid];
      if (first_nm.empty()) continue;
      char c = first_nm[0];
      if (c == 'b' || c == 'x') {
        GRBVar first_var = relax.getVar(first_vid);
        for (size_t j = 1; j < nodes.size(); ++j) {
          GRBVar second_var = relax.getVar(nodes[j]);
          relax.addConstr(first_var - second_var == 0);
        }
      }
    }

    relax.update();
    t0_section = now_sec();
    relax.optimize();
    t_relax += now_sec() - t0_section;

    // pull relaxation solution
    t0_section = now_sec();
    vector<double> x(numVars, 0.0);
    for (int i = 0; i < numVars; ++i) {
      try { x[i] = relax.getVar(i).get(GRB_DoubleAttr_X); } catch (...) { x[i] = 0.0; }
    }
    t_relax += now_sec() - t0_section;

    // score partitions and greedy rounding
    unordered_set<int> fixings;
    vector<pair<double,int>> scores;

    t0_section = now_sec();
    for (auto &p : var_partitions) {
      auto &nodes = p.second;
      if (nodes.size() <= 1) continue;
      double s = 0.0;
      for (int vid : nodes) s += var_obj[vid] * x[vid];
      scores.emplace_back(s, p.first);
    }
    sort(scores.begin(), scores.end(), greater<>());

    double threshold = 0.0;
    if (!scores.empty()) threshold = scores[0].first * 0.1;
    for (auto &sc : scores) {
      if (sc.first < threshold) break;
      int pid = sc.second;
      for (int vid : var_partitions[pid]) fixings.insert(vid);
    }

    vector<pair<double,int>> frac_vars;
    for (int i = 0; i < numVars; ++i) {
      if (fixings.count(i)) continue;
      if (x[i] > 1e-6 && x[i] < 0.9999) {
        frac_vars.emplace_back(var_obj[i] * (x[i] - 0.5), i);
      }
    }
    sort(frac_vars.begin(), frac_vars.end(), greater<>());

    int max_rounds = (int)frac_vars.size() / 3 + 2;
    for (int i = 0; i < (int)frac_vars.size() && i < max_rounds; ++i) {
      fixings.insert(frac_vars[i].second);
    }
    t_score += now_sec() - t0_section;

    vector<double> x2 = x;
    for (int vid : fixings) x2[vid] = 1.0;
    for (int i = 0; i < numVars; ++i) {
      if (!fixings.count(i) && x2[i] > 1e-6) x2[i] = (x2[i] > 0.5 ? 1.0 : 0.0);
    }

    // Stage model (original MIP with start)
    double stage1_obj = NAN;
    int stage1_status = -1;
    double stage1_runtime = 0.0;

    GRBModel full(model);
    full.set(GRB_IntParam_OutputFlag, VERBOSE ? 1 : 0);

    for (int i = 0; i < numVars; ++i) full.getVar(i).set(GRB_DoubleAttr_Start, x2[i]);
    full.update();

    // ---------------- STAGE 1: time-capped solve ----------------
    if (VERBOSE) {
      cout << "\n=== STAGE 1: time-capped solve (" << STAGE1_LIMIT << "s) with MIP start ===\n";
      cout.flush();
    }

    full.set(GRB_DoubleParam_TimeLimit, STAGE1_LIMIT);

    double t_stage1_start = now_sec();
    full.optimize();
    stage1_runtime = now_sec() - t_stage1_start;

    stage1_status = full.get(GRB_IntAttr_Status);
    if (full.get(GRB_IntAttr_SolCount) > 0) stage1_obj = full.get(GRB_DoubleAttr_ObjVal);

    double heuristic_obj = NAN;
    if (!is_nan(stage1_obj)) heuristic_obj = stage1_obj;
    else {
      heuristic_obj = 0.0;
      for (int i = 0; i < numVars; ++i) heuristic_obj += var_obj[i] * x2[i];
    }
    double heuristic_time = now_sec() - t0;

    // ---------------- STAGE 2: exact solve ----------------
    int final_status = stage1_status;
    double final_obj = NAN;
    double final_runtime = stage1_runtime;

    if (stage1_status != GRB_OPTIMAL) {
      if (VERBOSE) {
        cout << "\n=== STAGE 2: exact solve (no time limit) ===\n";
        cout.flush();
      }

      full.set(GRB_DoubleParam_TimeLimit, GRB_INFINITY);

      double t_stage2_start = now_sec();
      full.optimize();
      final_runtime = stage1_runtime + (now_sec() - t_stage2_start);
      final_status = full.get(GRB_IntAttr_Status);
    }

    if (full.get(GRB_IntAttr_SolCount) > 0) final_obj = full.get(GRB_DoubleAttr_ObjVal);

    double wall = now_sec() - t0;

    if (VERBOSE) {
      cout << "\n=== DONE ===\n";
      cout.flush();
    }

    // short timing breakdown to stderr
    cerr << "timings: graph=" << t_graph
         << " color=" << t_color
         << " relax=" << t_relax
         << " score=" << t_score
         << " stage1=" << stage1_runtime
         << " wall=" << wall << "\n";

    // JSON output
    ostringstream ss;
    ss << "{";
    ss << "\"tool\":\"heuristic_full_cpp\",";
    ss << "\"mps\":\"" << mps << "\",";
    ss << "\"verbose\":" << (VERBOSE ? "true" : "false") << ",";
    ss << "\"stage1_time_limit_sec\":" << STAGE1_LIMIT << ",";
    ss << "\"heuristic_obj\":" << (is_nan(heuristic_obj) ? string("null") : to_string(heuristic_obj)) << ",";
    ss << "\"heuristic_time_sec\":" << (is_nan(heuristic_time) ? string("null") : to_string(heuristic_time)) << ",";
    ss << "\"stage1_status\":" << stage1_status << ",";
    ss << "\"stage1_obj\":" << (is_nan(stage1_obj) ? string("null") : to_string(stage1_obj)) << ",";
    ss << "\"final_status\":" << final_status << ",";
    ss << "\"final_obj\":" << (is_nan(final_obj) ? string("null") : to_string(final_obj)) << ",";
    ss << "\"final_runtime_sec\":" << final_runtime << ",";
    ss << "\"wall_runtime_sec\":" << wall;
    ss << "}";

    cout << ss.str() << endl;

    delete[] vars;
    delete[] cons;
    delete[] qcons;

    return 0;
  }
  catch (GRBException &e) {
    cerr << "Gurobi error: " << e.getMessage() << "\n";
    return 2;
  }
  catch (exception &e) {
    cerr << "Error: " << e.what() << "\n";
    return 3;
  }
}