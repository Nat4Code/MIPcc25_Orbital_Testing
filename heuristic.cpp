// heuristic.cpp
// C++ port of qp_bench.py heuristic pipeline (lightweight, optimized for speed)
// (no final full solve: just compute the heuristic objective and time)

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

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: heuristic model.mps" << endl;
    return 1;
  }

  const string mps = argv[1];
  double t0 = now_sec();

  try {
    GRBEnv env(true);
    env.set(GRB_IntParam_LogToConsole, 0);
    env.start();

    GRBModel model(env, mps);
    model.set(GRB_IntParam_OutputFlag, 0);

    int numVars = model.get(GRB_IntAttr_NumVars);

    // gather variable info
    GRBVar* vars = model.getVars();
    unordered_map<string, int> var_index;
    vector<string> var_name(numVars);
    vector<char> var_vtype(numVars);
    vector<double> var_obj(numVars), var_lb(numVars), var_ub(numVars);
    for (int i = 0; i < numVars; ++i) {
      string nm = vars[i].get(GRB_StringAttr_VarName);
      var_name[i] = nm;
      var_index[nm] = i;
      var_vtype[i] = vars[i].get(GRB_CharAttr_VType);
      var_obj[i] = vars[i].get(GRB_DoubleAttr_Obj);
      var_lb[i] = vars[i].get(GRB_DoubleAttr_LB);
      var_ub[i] = vars[i].get(GRB_DoubleAttr_UB);
    }

    // build incidence graph
    unordered_map<string, unordered_map<string,double>> G;

    int numCons = model.get(GRB_IntAttr_NumConstrs);
    GRBConstr* cons = model.getConstrs();
    for (int ci = 0; ci < numCons; ++ci) {
      string cname = cons[ci].get(GRB_StringAttr_ConstrName);
      GRBLinExpr row = model.getRow(cons[ci]);
      int rsz = row.size();
      for (int j = 0; j < rsz; ++j) {
        string vnm = row.getVar(j).get(GRB_StringAttr_VarName);
        double a = row.getCoeff(j);
        G[vnm][cname] += fabs(a);
        G[cname][vnm] += fabs(a);
      }
    }

    int numQ = model.get(GRB_IntAttr_NumQConstrs);
    GRBQConstr* qcons = model.getQConstrs();
    for (int qi = 0; qi < numQ; ++qi) {
      string qname = string("q_cons") + to_string(qi);
      GRBQuadExpr qexpr = model.getQCRow(qcons[qi]);
      GRBLinExpr lin = qexpr.getLinExpr();
      int lsz = lin.size();
      for (int j = 0; j < lsz; ++j) {
        string vnm = lin.getVar(j).get(GRB_StringAttr_VarName);
        double a = lin.getCoeff(j);
        G[vnm][qname] += fabs(a);
        G[qname][vnm] += fabs(a);
      }
      int qsz = qexpr.size();
      for (int j = 0; j < qsz; ++j) {
        string v1 = qexpr.getVar1(j).get(GRB_StringAttr_VarName);
        string v2 = qexpr.getVar2(j).get(GRB_StringAttr_VarName);
        string tname = string("q_") + v1 + "_" + v2;
        double a = qexpr.getCoeff(j);
        G[tname][v1] += fabs(a);
        G[v1][tname] += fabs(a);
        G[tname][v2] += fabs(a);
        G[v2][tname] += fabs(a);
        G[tname][qname] += fabs(a);
        G[qname][tname] += fabs(a);
      }
    }

    // initial coloring
    unordered_map<string, vector<string>> color_classes;
    unordered_map<string,int> node_to_color;
    int color_id = 0;

    for (int ci = 0; ci < numCons; ++ci) {
      string cname = cons[ci].get(GRB_StringAttr_ConstrName);
      double rhs = cons[ci].get(GRB_DoubleAttr_RHS);
      char sense = cons[ci].get(GRB_CharAttr_Sense);
      string key = string("C|") + sense + "|" + to_string((long long)llround(rhs*1e9));
      if (!color_classes.count(key)) color_classes[key] = {};
      color_classes[key].push_back(cname);
    }
    for (int qi = 0; qi < numQ; ++qi) {
      string cname = string("q_cons") + to_string(qi);
      double rhs = qcons[qi].get(GRB_DoubleAttr_QCRHS);
      char sense = qcons[qi].get(GRB_CharAttr_QCSense);
      string key = string("Q|") + sense + "|" + to_string((long long)llround(rhs*1e9));
      if (!color_classes.count(key)) color_classes[key] = {};
      color_classes[key].push_back(cname);
    }
    for (int i = 0; i < numVars; ++i) {
      string key = string("V|") + var_vtype[i] + "|obj:" + to_string((long long)llround(var_obj[i]*1e9))
        + "|lb:" + to_string((long long)llround(var_lb[i]*1e9)) + "|ub:" + to_string((long long)llround(var_ub[i]*1e9));
      if (!color_classes.count(key)) color_classes[key] = {};
      color_classes[key].push_back(var_name[i]);
    }
    for (auto &p : color_classes) {
      int cid = color_id++;
      for (auto &n : p.second) node_to_color[n] = cid;
    }
    unordered_map<int, vector<string>> new_classes;
    for (auto &p : color_classes) {
      unordered_map<long long, vector<string>> tmp;
      for (auto &n : p.second) {
        long long deg = 0;
        for (auto &kv : G[n]) deg += (long long)llround(kv.second*1000.0);
        tmp[deg].push_back(n);
      }
      for (auto &q : tmp) {
        int cid = color_id++;
        for (auto &n : q.second) new_classes[cid].push_back(n);
      }
    }
    unordered_map<int, vector<string>> var_partitions;
    for (auto &c : new_classes) {
      for (auto &n : c.second) {
        if (!n.empty() && (n[0] == 'b' || n[0] == 'x' || var_index.count(n))) {
          var_partitions[c.first].push_back(n);
        }
      }
    }

    // relaxation and greedy fixings
    GRBModel relax = model.relax();
    relax.set(GRB_IntParam_OutputFlag, 0);
    relax.update();
    relax.optimize();

    vector<double> x(numVars, 0.0);
    for (int i = 0; i < numVars; ++i) {
      try { x[i] = relax.getVar(i).get(GRB_DoubleAttr_X); } catch (...) { x[i] = 0.0; }
    }

    double tol = 1e-5;
    unordered_set<string> fixed0;
    for (int i = 0; i < numVars; ++i) {
      if (x[i] <= tol) {
        fixed0.insert(var_name[i]);
      }
    }

    unordered_set<string> fixings;
    vector<unordered_set<string>> history;
    int max_iters = 50;
    for (int it = 0; it < max_iters; ++it) {
      double best_score = 1.1;
      int best_pid = -1;
      for (auto &p : var_partitions) {
        auto &nodes = p.second;
        if (nodes.size() <= 1) continue;
        double s = 0.0;
        for (auto &nm : nodes) {
          auto itv = var_index.find(nm);
          if (itv == var_index.end()) continue;
          s += x[itv->second];
        }
        double score = s;
        if (score > best_score) { best_score = score; best_pid = p.first; }
      }
      if (best_pid < 0) break;
      for (auto &nm : var_partitions[best_pid]) fixings.insert(nm);
      history.push_back(fixings);

      GRBModel cur_relax = model.relax();
      for (auto &nm : fixings) {
        auto itv = var_index.find(nm);
        if (itv == var_index.end()) continue;
        int idx = itv->second;
        cur_relax.getVar(idx).set(GRB_DoubleAttr_LB, 1.0);
      }
      for (auto &nm : fixed0) {
        auto itv = var_index.find(nm);
        if (itv == var_index.end()) continue;
        int idx = itv->second;
        cur_relax.getVar(idx).set(GRB_DoubleAttr_UB, 0.0);
      }
      cur_relax.update();
      cur_relax.optimize();

      int status = cur_relax.get(GRB_IntAttr_Status);
      if (status != GRB_OPTIMAL && status != GRB_SUBOPTIMAL) {
        fixings = history.size() > 1 ? history[history.size()-2] : unordered_set<string>();
        break;
      }

      for (int i = 0; i < numVars; ++i) {
        try { x[i] = cur_relax.getVar(i).get(GRB_DoubleAttr_X); } catch (...) { }
      }
    }

    // compute heuristic objective using last relaxation solution (no Gurobi solve)
    double heuristic_obj = NAN;
    double heuristic_time = NAN;
    if (!x.empty()) {
      heuristic_obj = 0.0;
      for (int i = 0; i < numVars; ++i) {
        heuristic_obj += var_obj[i] * x[i];
      }
      heuristic_time = now_sec() - t0;
    } else {
      fixings.clear();
    }

    double wall = now_sec() - t0;

    ostringstream ss;
    ss << "{\"tool\":\"heuristic_cpp\",\"mps\":\"" << mps << "\",";
    ss << "\"heuristic_obj\":" << (std::isnan(heuristic_obj) ? string("null") : to_string(heuristic_obj)) << ",";
    ss << "\"heuristic_time_sec\":" << (std::isnan(heuristic_time) ? string("null") : to_string(heuristic_time)) << ",";
    ss << "\"wall_runtime_sec\":" << wall << "}";
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
