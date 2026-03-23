// heuristic_fast.cpp
// Speed-optimized pure heuristic implementation (index-based, minimal overhead)
//  * one LP relaxation
//  * generate integer partitions once
//  * score partitions using variable indices
//  * pick top-K partitions without rebuilding models
//  * avoid string maps in inner loops
//  * purely compute objective from modified relaxation x

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
    cerr << "Usage: heuristic_simple model.mps" << endl;
    return 1;
  }

  const string mps = argv[1];
  double t0 = now_sec();
  double t_graph=0, t_color=0, t_relax=0, t_score=0;

  try {
    GRBEnv env(true);
    env.set(GRB_IntParam_LogToConsole, 0);
    env.start();

    GRBModel model(env, mps);
    model.set(GRB_IntParam_OutputFlag, 0);

    int numVars = model.get(GRB_IntAttr_NumVars);

    // gather variable info (use indices throughout)
    GRBVar* vars = model.getVars();
    vector<string> var_name(numVars);
    vector<char> var_vtype(numVars);
    vector<double> var_obj(numVars), var_lb(numVars), var_ub(numVars);
    unordered_map<string,int> idx_by_name;
    idx_by_name.reserve(numVars*2);
    for (int i = 0; i < numVars; ++i) {
      string nm = vars[i].get(GRB_StringAttr_VarName);
      var_name[i] = nm;
      idx_by_name[nm] = i;
      var_vtype[i] = vars[i].get(GRB_CharAttr_VType);
      var_obj[i] = vars[i].get(GRB_DoubleAttr_Obj);
      var_lb[i] = vars[i].get(GRB_DoubleAttr_LB);
      var_ub[i] = vars[i].get(GRB_DoubleAttr_UB);
    }

    // build incidence graph
    // build incidence graph using string identifiers for nodes
    unordered_map<string, unordered_map<string,double>> G;

    int numCons = model.get(GRB_IntAttr_NumConstrs);
    GRBConstr* cons = model.getConstrs();
    double t0_section = now_sec();
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
    t_graph += now_sec() - t0_section;

    int numQ = model.get(GRB_IntAttr_NumQConstrs);
    GRBQConstr* qcons = model.getQConstrs();
    t0_section = now_sec();
    for (int qi = 0; qi < numQ; ++qi) {
      // represent quad constraint as a string node
      string qname = string("Q|") + to_string(qi);
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
        // quadratic terms already contribute via linear part, so nothing to add
        string v1 = qexpr.getVar1(j).get(GRB_StringAttr_VarName);
        string v2 = qexpr.getVar2(j).get(GRB_StringAttr_VarName);
        (void)v1; (void)v2;
      }
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
    t0_section = now_sec();
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
    t_color += now_sec() - t0_section;
    // convert variable partitions to index lists for faster loops
    unordered_map<int, vector<int>> var_partitions;
    for (auto &c : new_classes) {
      for (auto &n : c.second) {
        if (!n.empty()) {
          auto it = idx_by_name.find(n);
          if (it != idx_by_name.end()) {
            var_partitions[c.first].push_back(it->second);
          }
        }
      }
    }

    // relaxation
    GRBModel relax = model.relax();
    relax.set(GRB_IntParam_OutputFlag, 0);

    // add folding constraints to the relaxation based on partitions
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

    t0_section = now_sec();
    vector<double> x(numVars, 0.0);
    for (int i = 0; i < numVars; ++i) {
      try { x[i] = relax.getVar(i).get(GRB_DoubleAttr_X); } catch (...) { x[i] = 0.0; }
    }
    t_relax += now_sec() - t0_section;

    // FAST HEURISTIC: score partitions by objective contribution and greedy rounding
    unordered_set<int> fixings;
    vector<pair<double,int>> scores; // (objective_contribution, partition_id)
    t0_section = now_sec();
    
    // Score partitions by objective contribution (sum of obj_coeff * x_value)
    for (auto &p : var_partitions) {
      auto &nodes = p.second;
      if (nodes.size() <= 1) continue;
      double s = 0.0;
      for (int vid : nodes) {
        s += var_obj[vid] * x[vid];
      }
      scores.emplace_back(s, p.first);
    }
    
    // Sort by objective contribution (descending) and fix high-impact partitions
    sort(scores.begin(), scores.end(), greater<>());
    double threshold = 0.0;
    if (!scores.empty()) {
      threshold = scores[0].first * 0.1;  // fix partitions with 10% of max contribution
    }
    for (auto &sc : scores) {
      if (sc.first < threshold) break;
      int pid = sc.second;
      for (int vid : var_partitions[pid]) fixings.insert(vid);
    }
    
    // Greedy rounding: round fractional variables based on objective contribution
    vector<pair<double, int>> frac_vars; // (objective_coeff * x, var_id)
    for (int i = 0; i < numVars; ++i) {
      if (fixings.count(i)) continue;  // skip already-fixed vars
      if (x[i] > 1e-6 && x[i] < 0.9999) {  // fractional
        frac_vars.emplace_back(var_obj[i] * (x[i] - 0.5), i);  // score for rounding
      }
    }
    sort(frac_vars.begin(), frac_vars.end(), greater<>());
    
    // Greedily round high-value fractional variables
    int max_rounds = (int)frac_vars.size() / 3 + 2;  // round top third or minimum 2
    for (int i = 0; i < (int)frac_vars.size() && i < max_rounds; ++i) {
      int vid = frac_vars[i].second;
      fixings.insert(vid);
    }
    
    t_score += now_sec() - t0_section;

    // Build solution: set fixed vars to 1, otherwise use LP values rounded to nearest
    vector<double> x2 = x;
    for (int vid : fixings) {
      x2[vid] = 1.0;
    }
    
    // Round remaining fractional variables to nearest integer for better feasibility
    for (int i = 0; i < numVars; ++i) {
      if (!fixings.count(i) && x2[i] > 1e-6) {
        if (x2[i] > 0.5) {
          x2[i] = 1.0;
        } else {
          x2[i] = 0.0;
        }
      }
    }

    double heuristic_obj = 0.0;
    for (int i = 0; i < numVars; ++i) {
      heuristic_obj += var_obj[i] * x2[i];
    }
    double heuristic_time = now_sec() - t0;

    double wall = now_sec() - t0;
    cerr << "timings: graph=" << t_graph << " color=" << t_color << " relax=" << t_relax << " score=" << t_score << "\n";

    ostringstream ss;
    ss << "{\"tool\":\"heuristic_simple_cpp\",\"mps\":\"" << mps << "\",";
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
