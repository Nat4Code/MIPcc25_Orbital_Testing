/////////////////////////////////////////////////////////////////////////////////
// orbit_branch_depth_bliss.cpp
//
// Depth-limited custom (orbital) branching using orbit_helper (bliss in helper).
//
// OPTION A IMPLEMENTED:
//   - We DO NOT hand off to Gurobi merely because the LP relaxation is integral.
//   - Instead, if LP is integral on binaries, we treat it as a FEASIBLE incumbent
//     (since it satisfies all constraints and binaries are integral) and stop.
//   - Handoff to Gurobi happens ONLY when node.depth >= orbit_depth.
//
// Notes:
//   - If the instance has 0 constraints, the root LP is integral and this will
//     finish immediately (fast) with an incumbent, rather than running a full MIP.
//
// CLI:
//   ./orbit_branch_depth model.mps [--time-limit T] [--node-limit N]
//                                 [--threads K] [--gurobi-threads G]
//                                 [--sym-budget S] [--orbit-depth D]
//
// Defaults:
//   --threads (workers)        = 16
//   --gurobi-threads per worker= 0   (0 => do not set Threads; let Gurobi decide)
//   --sym-budget               = 0.05 seconds
//   --orbit-depth              = 3
/////////////////////////////////////////////////////////////////////////////////

#include <gurobi_c++.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <mutex>
#include <condition_variable>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>
#include <thread>

extern char **environ;

using json = nlohmann::json;
using std::string;

static inline double wall_sec_since(const std::chrono::steady_clock::time_point& t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// ----------------------- crash-safe checkpoints -----------------------
static std::atomic<const char*> g_checkpoint{"init"};
static const char* g_mps_cstr = nullptr;

static void emit_crash_json(const char* sig_name) {
  json out;
  out["tool"] = "orbit_branch_depth_bliss";
  out["ok"] = false;
  out["signal"] = sig_name;
  out["checkpoint"] = g_checkpoint.load();
  out["mps"] = (g_mps_cstr ? string(g_mps_cstr) : "");
  try { std::cout << out.dump() << "\n"; std::cout.flush(); } catch (...) {}
}

static void signal_handler(int sig) {
  if (sig == SIGSEGV) emit_crash_json("SIGSEGV");
  else if (sig == SIGABRT) emit_crash_json("SIGABRT");
  else emit_crash_json("SIGNAL");
  std::_Exit(128 + sig);
}

static inline void set_checkpoint(const char* s) {
  g_checkpoint.store(s, std::memory_order_relaxed);
}

// --------------------------- numeric canonicalization ---------------------------
static inline long long qkey(double x) {
  if (std::isnan(x)) return (long long)0x7ff8000000000000ULL;
  if (x ==  std::numeric_limits<double>::infinity())  return (long long)0x7ff0000000000000ULL;
  if (x == -std::numeric_limits<double>::infinity())  return (long long)0xfff0000000000000ULL;

  const double SCALE = 1e9;
  double y = x * SCALE;

  if (y > (double)LLONG_MAX) return LLONG_MAX;
  if (y < (double)LLONG_MIN) return LLONG_MIN;
  return llround(y);
}

struct ColorPool {
  std::unordered_map<std::string, unsigned> mp;
  unsigned next = 1;

  unsigned get(const std::string& k) {
    auto it = mp.find(k);
    if (it != mp.end()) return it->second;
    unsigned c = next++;
    mp.emplace(k, c);
    return c;
  }
};

static inline std::string quad_term_key(const std::string& v1, const std::string& v2) {
  std::string a = v1, b = v2;
  if (a > b) std::swap(a, b);
  return "q(" + a + "," + b + ")";
}

// --------------------------- built graph struct ---------------------------
struct BuiltGraph {
  std::vector<unsigned> color;
  std::vector<std::vector<unsigned>> adj;
  std::vector<std::pair<unsigned, unsigned>> edges;

  std::vector<unsigned> var_vertex_of_index;

  std::vector<int> bin_var_indices;
  std::vector<unsigned> bin_var_vertex_ids;

  unsigned add_vertex(unsigned col) {
    unsigned id = (unsigned)color.size();
    color.push_back(col);
    adj.emplace_back();
    return id;
  }

  void add_edge(unsigned u, unsigned v) {
    if (u == v) return;
    if (u >= adj.size() || v >= adj.size()) throw std::runtime_error("add_edge: id out of range");
    adj[u].push_back(v);
    adj[v].push_back(u);
  }

  unsigned n() const { return (unsigned)color.size(); }

  void dedup_and_build_edgelist() {
    for (unsigned u = 0; u < n(); ++u) {
      auto& nbr = adj[u];
      std::sort(nbr.begin(), nbr.end());
      nbr.erase(std::unique(nbr.begin(), nbr.end()), nbr.end());
    }
    edges.clear();
    edges.reserve(n() * 2);
    for (unsigned u = 0; u < n(); ++u) {
      for (unsigned v : adj[u]) if (u < v) edges.emplace_back(u, v);
    }
  }
};

static void validate_BG(const BuiltGraph& BG) {
  const unsigned N = BG.n();
  for (unsigned u = 0; u < N; ++u) {
    for (unsigned v : BG.adj[u]) {
      if (v >= N) throw std::runtime_error("BG.adj out-of-range neighbor");
      if (u == v) throw std::runtime_error("BG.adj self-loop");
    }
  }
  for (auto &e : BG.edges) {
    if (e.first >= N || e.second >= N || e.first == e.second)
      throw std::runtime_error("BG.edges invalid endpoint");
  }
}

// --------------------------- build exact graph from gurobi ---------------------------
static BuiltGraph build_exact_graph_from_gurobi(GRBModel& model,
                                               int& numVarsOut,
                                               int& numBinOut,
                                               int& numConstrOut,
                                               int& numQConsOut) {
  set_checkpoint("build_exact_graph:begin");

  BuiltGraph graph;
  ColorPool colors;

  const int numVars    = model.get(GRB_IntAttr_NumVars);
  const int numConstrs = model.get(GRB_IntAttr_NumConstrs);
  const int numQCons   = model.get(GRB_IntAttr_NumQConstrs);

  numVarsOut   = numVars;
  numConstrOut = numConstrs;
  numQConsOut  = numQCons;

  GRBVar* vars      = (numVars > 0)    ? model.getVars()     : nullptr;
  GRBConstr* cons   = (numConstrs > 0) ? model.getConstrs()  : nullptr;
  GRBQConstr* qcons = (numQCons > 0)   ? model.getQConstrs() : nullptr;

  graph.var_vertex_of_index.assign((size_t)numVars, 0);

  // variables
  set_checkpoint("build_exact_graph:vars");
  numBinOut = 0;

  graph.bin_var_indices.clear();
  graph.bin_var_vertex_ids.clear();
  graph.bin_var_indices.reserve((size_t)numVars);
  graph.bin_var_vertex_ids.reserve((size_t)numVars);

  for (int var_idx = 0; var_idx < numVars; ++var_idx) {
    char vtype = vars[var_idx].get(GRB_CharAttr_VType);
    double obj = vars[var_idx].get(GRB_DoubleAttr_Obj);
    double lb  = vars[var_idx].get(GRB_DoubleAttr_LB);
    double ub  = vars[var_idx].get(GRB_DoubleAttr_UB);

    std::string key = "V|";
    key.push_back(vtype);
    key += "|obj:" + std::to_string(qkey(obj));
    key += "|lb:"  + std::to_string(qkey(lb));
    key += "|ub:"  + std::to_string(qkey(ub));

    unsigned var_vertex_id = graph.add_vertex(colors.get(key));
    graph.var_vertex_of_index[(size_t)var_idx] = var_vertex_id;

    if (vtype == GRB_BINARY) {
      ++numBinOut;
      graph.bin_var_indices.push_back(var_idx);
      graph.bin_var_vertex_ids.push_back(var_vertex_id);
    }
  }

  // name map
  set_checkpoint("build_exact_graph:name_map");
  std::unordered_map<std::string, unsigned> name2varvid;
  name2varvid.reserve((size_t)numVars * 2);
  for (int var_idx = 0; var_idx < numVars; ++var_idx) {
    name2varvid.emplace(vars[var_idx].get(GRB_StringAttr_VarName),
                        graph.var_vertex_of_index[(size_t)var_idx]);
  }

  // linear constraints
  set_checkpoint("build_exact_graph:lin_cons");
  for (int ci = 0; ci < numConstrs; ++ci) {
    double rhs = cons[ci].get(GRB_DoubleAttr_RHS);
    char sense = cons[ci].get(GRB_CharAttr_Sense);

    std::string ckey = "C|";
    ckey.push_back(sense);
    ckey += "|rhs:" + std::to_string(qkey(rhs));
    unsigned con_vertex_id = graph.add_vertex(colors.get(ckey));

    GRBLinExpr row = model.getRow(cons[ci]);
    int row_nz = row.size();
    for (int j = 0; j < row_nz; ++j) {
      GRBVar v = row.getVar(j);
      double a = row.getCoeff(j);
      std::string vname = v.get(GRB_StringAttr_VarName);

      auto it = name2varvid.find(vname);
      if (it == name2varvid.end()) continue;
      unsigned var_vertex_id = it->second;

      unsigned coeff_vertex_id = graph.add_vertex(colors.get("A|lin|" + std::to_string(qkey(a))));
      graph.add_edge(con_vertex_id, coeff_vertex_id);
      graph.add_edge(coeff_vertex_id, var_vertex_id);
    }
  }

  // quadratic constraint vertices
  set_checkpoint("build_exact_graph:quad_cons_vertices");
  std::vector<unsigned> qcon_vertex_id((size_t)numQCons, 0);
  for (int qi = 0; qi < numQCons; ++qi) {
    double rhs = qcons[qi].get(GRB_DoubleAttr_QCRHS);
    char sense = qcons[qi].get(GRB_CharAttr_QCSense); // correct attribute type

    std::string qkey_s = "Q|";
    qkey_s.push_back(sense);
    qkey_s += "|rhs:" + std::to_string(qkey(rhs));
    qcon_vertex_id[(size_t)qi] = graph.add_vertex(colors.get(qkey_s));
  }

  // quadratic constraint content
  set_checkpoint("build_exact_graph:quad_content");
  std::unordered_map<std::string, unsigned> term2vid;
  term2vid.reserve((size_t)numQCons * 4);

  for (int qi = 0; qi < numQCons; ++qi) {
    unsigned qvertex = qcon_vertex_id[(size_t)qi];
    GRBQuadExpr qexpr = model.getQCRow(qcons[qi]);

    // linear part
    {
      GRBLinExpr lin = qexpr.getLinExpr();
      int lsz = lin.size();
      for (int j = 0; j < lsz; ++j) {
        GRBVar v = lin.getVar(j);
        double a = lin.getCoeff(j);
        std::string vname = v.get(GRB_StringAttr_VarName);

        auto it = name2varvid.find(vname);
        if (it == name2varvid.end()) continue;
        unsigned var_vertex_id = it->second;

        unsigned coeff_vertex_id = graph.add_vertex(colors.get("A|qclin|" + std::to_string(qkey(a))));
        graph.add_edge(qvertex, coeff_vertex_id);
        graph.add_edge(coeff_vertex_id, var_vertex_id);
      }
    }

    // quadratic part
    int qsz = qexpr.size();
    for (int j = 0; j < qsz; ++j) {
      GRBVar v1 = qexpr.getVar1(j);
      GRBVar v2 = qexpr.getVar2(j);
      double a  = qexpr.getCoeff(j);

      std::string n1 = v1.get(GRB_StringAttr_VarName);
      std::string n2 = v2.get(GRB_StringAttr_VarName);

      auto it1 = name2varvid.find(n1);
      auto it2 = name2varvid.find(n2);
      if (it1 == name2varvid.end() || it2 == name2varvid.end()) continue;

      unsigned var_vertex_1 = it1->second;
      unsigned var_vertex_2 = it2->second;

      std::string tkey = quad_term_key(n1, n2);
      unsigned term_vertex_id;
      auto itt = term2vid.find(tkey);
      if (itt == term2vid.end()) {
        term_vertex_id = graph.add_vertex(colors.get("T|term"));
        term2vid.emplace(tkey, term_vertex_id);
        graph.add_edge(term_vertex_id, var_vertex_1);
        graph.add_edge(term_vertex_id, var_vertex_2);
      } else {
        term_vertex_id = itt->second;
      }

      unsigned coeff_vertex_id = graph.add_vertex(colors.get("A|qcquad|" + std::to_string(qkey(a))));
      graph.add_edge(qvertex, coeff_vertex_id);
      graph.add_edge(coeff_vertex_id, term_vertex_id);
    }
  }

  if (vars)  delete[] vars;
  if (cons)  delete[] cons;
  if (qcons) delete[] qcons;

  set_checkpoint("build_exact_graph:dedup");
  graph.dedup_and_build_edgelist();

  set_checkpoint("build_exact_graph:done");
  return graph;
}

// -------------------- memfd graph serialization for helper --------------------
static bool write_all_fd(int fd, const void* buf, size_t n) {
  const char* p = (const char*)buf;
  size_t off = 0;
  while (off < n) {
    ssize_t w = ::write(fd, p + off, n - off);
    if (w <= 0) return false;
    off += (size_t)w;
  }
  return true;
}

// format must match orbit_helper load_graph_from_fd:
// magic 'OBBG' (0x4f424247), ver=1
// N, numBin, base_color[N], bin_vertex_ids[numBin], m, edges[m] as u32,u32
static int create_bg_memfd_and_write(const BuiltGraph& graph) {
  int fd = (int)syscall(SYS_memfd_create, "obbg", 0);
  if (fd < 0) throw std::runtime_error("memfd_create failed");

  int flags = fcntl(fd, F_GETFD);
  if (flags >= 0) fcntl(fd, F_SETFD, flags & ~FD_CLOEXEC);

  uint32_t magic = 0x4f424247u;
  uint32_t ver   = 1u;
  uint32_t N     = (uint32_t)graph.n();
  uint32_t numBin = (uint32_t)graph.bin_var_vertex_ids.size();
  uint32_t m = (uint32_t)graph.edges.size();

  if (!write_all_fd(fd, &magic, sizeof(magic))) throw std::runtime_error("write memfd failed");
  if (!write_all_fd(fd, &ver, sizeof(ver))) throw std::runtime_error("write memfd failed");
  if (!write_all_fd(fd, &N, sizeof(N))) throw std::runtime_error("write memfd failed");
  if (!write_all_fd(fd, &numBin, sizeof(numBin))) throw std::runtime_error("write memfd failed");

  std::vector<uint32_t> base_color(N);
  for (uint32_t i = 0; i < N; ++i) base_color[i] = (uint32_t)graph.color[i];
  if (N > 0 && !write_all_fd(fd, base_color.data(), (size_t)N * sizeof(uint32_t)))
    throw std::runtime_error("write memfd failed");

  std::vector<uint32_t> bin_vid(numBin);
  for (uint32_t i = 0; i < numBin; ++i) bin_vid[i] = (uint32_t)graph.bin_var_vertex_ids[i];
  if (numBin > 0 && !write_all_fd(fd, bin_vid.data(), (size_t)numBin * sizeof(uint32_t)))
    throw std::runtime_error("write memfd failed");

  if (!write_all_fd(fd, &m, sizeof(m))) throw std::runtime_error("write memfd failed");
  for (uint32_t i = 0; i < m; ++i) {
    uint32_t u = (uint32_t)graph.edges[i].first;
    uint32_t v = (uint32_t)graph.edges[i].second;
    if (!write_all_fd(fd, &u, sizeof(u))) throw std::runtime_error("write memfd failed");
    if (!write_all_fd(fd, &v, sizeof(v))) throw std::runtime_error("write memfd failed");
  }

  ::lseek(fd, 0, SEEK_SET);
  return fd;
}

// -------------------- Helper session --------------------
struct HelperSession {
  int bg_fd = -1;

  pid_t pid = -1;
  int wfd = -1;
  int rfd = -1;

  int failures = 0;
  bool disabled = false;

  void note_failure_and_maybe_disable() {
    failures++;
    if (failures >= 3) disabled = true;
  }

  void stop(bool count_failure = false) {
    if (count_failure) note_failure_and_maybe_disable();

    if (wfd >= 0) { close(wfd); wfd = -1; }
    if (rfd >= 0) { close(rfd); rfd = -1; }

    if (pid > 0) {
      int st = 0;
      kill(pid, SIGKILL);
      waitpid(pid, &st, 0);
      pid = -1;
    }
  }

  bool alive() const {
    if (pid <= 0) return false;
    int st = 0;
    pid_t w = waitpid(pid, &st, WNOHANG);
    return (w == 0);
  }

  bool start() {
    stop(false);
    if (disabled) return false;

    int inpipe[2], outpipe[2];
    if (pipe(inpipe) != 0) return false;
    if (pipe(outpipe) != 0) {
      close(inpipe[0]); close(inpipe[1]);
      return false;
    }

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);

    posix_spawn_file_actions_adddup2(&fa, inpipe[0], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&fa, outpipe[1], STDOUT_FILENO);

    int devnull = ::open("/dev/null", O_WRONLY);
    if (devnull >= 0) {
      posix_spawn_file_actions_adddup2(&fa, devnull, STDERR_FILENO);
      posix_spawn_file_actions_addclose(&fa, devnull);
    }

    posix_spawn_file_actions_addclose(&fa, inpipe[1]);
    posix_spawn_file_actions_addclose(&fa, outpipe[0]);

    std::string fdstr = std::to_string(bg_fd);
    char* argv[] = { (char*)"./orbit_helper", (char*)"--bg-fd", (char*)fdstr.c_str(), nullptr };

    int rc = posix_spawn(&pid, "./orbit_helper", &fa, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&fa);

    close(inpipe[0]);
    close(outpipe[1]);

    if (rc != 0) {
      close(inpipe[1]);
      close(outpipe[0]);
      pid = -1;
      note_failure_and_maybe_disable();
      return false;
    }

    wfd = inpipe[1];
    rfd = outpipe[0];
    return true;
  }

  bool ensure_alive() {
    if (disabled) return false;
    if (alive()) return true;
    return start();
  }
};

static bool read_all_fd(int fd, void* buf, size_t n) {
  char* p = (char*)buf;
  size_t off = 0;
  while (off < n) {
    ssize_t r = ::read(fd, p + off, n - off);
    if (r <= 0) return false;
    off += (size_t)r;
  }
  return true;
}

static bool session_orbit_query(HelperSession& session,
                               uint32_t seed_vertex,
                               const std::vector<int8_t>& fix_by_bin,
                               double time_limit_sec,
                               bool& terminated,
                               std::vector<uint32_t>& orbit_vertices_out) {
  terminated = false;
  orbit_vertices_out.clear();

  if (!session.ensure_alive()) return false;

  uint8_t req_type = 0;
  if (!write_all_fd(session.wfd, &req_type, sizeof(req_type))) { session.stop(true); return false; }
  if (!write_all_fd(session.wfd, &seed_vertex, sizeof(seed_vertex))) { session.stop(true); return false; }
  if (!write_all_fd(session.wfd, &time_limit_sec, sizeof(time_limit_sec))) { session.stop(true); return false; }
  if (!fix_by_bin.empty() &&
      !write_all_fd(session.wfd, fix_by_bin.data(), fix_by_bin.size() * sizeof(int8_t))) {
    session.stop(true);
    return false;
  }

  uint8_t ok = 0, term = 0;
  if (!read_all_fd(session.rfd, &ok, sizeof(ok)) ||
      !read_all_fd(session.rfd, &term, sizeof(term))) {
    session.stop(true);
    return false;
  }

  terminated = (term == 1);

  if (ok != 1) {
    session.note_failure_and_maybe_disable();
    return false;
  }

  uint32_t k = 0;
  if (!read_all_fd(session.rfd, &k, sizeof(k))) { session.stop(true); return false; }
  orbit_vertices_out.resize(k);
  if (k > 0 && !read_all_fd(session.rfd, orbit_vertices_out.data(), (size_t)k * sizeof(uint32_t))) {
    session.stop(true);
    orbit_vertices_out.clear();
    return false;
  }

  return true;
}

// --------------------------- branching helpers ---------------------------
static int pick_fractional_binary_pos(const std::vector<double>& x,
                                      const std::vector<int>& bin_var_indices,
                                      double tol = 1e-6) {
  int best_pos = -1;
  double best_frac = 0.0;
  for (int pos = 0; pos < (int)bin_var_indices.size(); ++pos) {
    int var_idx = bin_var_indices[(size_t)pos];
    double v = x[(size_t)var_idx];
    double frac = std::fabs(v - std::round(v));
    if (frac > tol && frac > best_frac) {
      best_frac = frac;
      best_pos = pos;
    }
  }
  return best_pos;
}

struct Node {
  std::vector<int8_t> fix_bin; // -1 unset, 0/1 fixed; size = numBin
  int depth = 0;
};

struct WorkStack {
  std::mutex m;
  std::condition_variable cv;
  std::vector<Node> st;

  bool done = false;
  int active = 0;

  bool pop(Node& out) {
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [&](){ return done || !st.empty(); });
    if (done && st.empty()) return false;

    out = std::move(st.back());
    st.pop_back();
    active++;
    return true;
  }

  void push(Node&& n) {
    {
      std::lock_guard<std::mutex> lk(m);
      st.push_back(std::move(n));
    }
    cv.notify_one();
  }

  void worker_finished_unit() {
    std::lock_guard<std::mutex> lk(m);
    active--;
    if (st.empty() && active == 0) {
      done = true;
      cv.notify_all();
    }
  }

  void stop_all() {
    {
      std::lock_guard<std::mutex> lk(m);
      done = true;
    }
    cv.notify_all();
  }
};

// --------------------------- sub-MIP solve at depth cutoff ---------------------------
static void solve_submip_with_fixings(GRBModel& base_mip_model,
                                     const BuiltGraph& graph,
                                     const Node& node,
                                     int numVars,
                                     int grb_threads,
                                     double remaining_time,
                                     bool have_inc,
                                     double best_obj,
                                     std::mutex& inc_mtx,
                                     bool& have_inc_ref,
                                     double& best_obj_ref,
                                     std::vector<double>& best_sol_ref,
                                     std::atomic<bool>& terminated_flag) {
  GRBModel mip(base_mip_model);
  mip.set(GRB_IntParam_OutputFlag, 0);
  if (grb_threads > 0) mip.set(GRB_IntParam_Threads, grb_threads);
  if (remaining_time > 0.0) mip.set(GRB_DoubleParam_TimeLimit, remaining_time);

  if (have_inc) {
    mip.set(GRB_DoubleParam_Cutoff, best_obj - 1e-9);
  }

  GRBVar* mvars_ptr = mip.getVars();
  std::vector<GRBVar> mvars((size_t)numVars);
  for (int i = 0; i < numVars; ++i) mvars[(size_t)i] = mvars_ptr[i];
  delete[] mvars_ptr;

  for (size_t bpos = 0; bpos < graph.bin_var_indices.size(); ++bpos) {
    int8_t f = node.fix_bin[bpos];
    if (f < 0) continue;
    int var_idx = graph.bin_var_indices[bpos];
    if (f == 0) {
      mvars[(size_t)var_idx].set(GRB_DoubleAttr_LB, 0.0);
      mvars[(size_t)var_idx].set(GRB_DoubleAttr_UB, 0.0);
    } else {
      mvars[(size_t)var_idx].set(GRB_DoubleAttr_LB, 1.0);
      mvars[(size_t)var_idx].set(GRB_DoubleAttr_UB, 1.0);
    }
  }
  mip.update();

  mip.optimize();

  int st = mip.get(GRB_IntAttr_Status);
  if (st == GRB_TIME_LIMIT) {
    terminated_flag.store(true);
  }

  int solcnt = mip.get(GRB_IntAttr_SolCount);
  if (solcnt > 0) {
    double obj = mip.get(GRB_DoubleAttr_ObjVal);

    std::vector<double> x((size_t)numVars, 0.0);
    for (int i = 0; i < numVars; ++i) x[(size_t)i] = mvars[(size_t)i].get(GRB_DoubleAttr_X);

    std::lock_guard<std::mutex> lk(inc_mtx);
    if (!have_inc_ref || obj < best_obj_ref) {
      have_inc_ref = true;
      best_obj_ref = obj;
      best_sol_ref = std::move(x);
    }
  }
}

// --------------------------- main ---------------------------
int main(int argc, char** argv) {
#if !defined(__SANITIZE_ADDRESS__) && !defined(__SANITIZE_UNDEFINED__)
  std::signal(SIGSEGV, signal_handler);
  std::signal(SIGABRT, signal_handler);
#endif

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " model.mps [--time-limit T] [--node-limit N] [--threads K]"
                 " [--gurobi-threads G] [--sym-budget S] [--orbit-depth D]\n";
    return 1;
  }

  const std::string mps = argv[1];
  g_mps_cstr = argv[1];

  double time_limit = 0.0;
  int node_limit = 20000;

  int workers = 16;
  int grb_threads = 0;      // 0 => do not set Threads
  double sym_budget = 0.05;
  int orbit_depth = 3;

  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--time-limit" && i+1 < argc) time_limit = std::stod(argv[++i]);
    else if (a == "--node-limit" && i+1 < argc) node_limit = std::stoi(argv[++i]);
    else if (a == "--threads" && i+1 < argc) workers = std::stoi(argv[++i]);
    else if (a == "--gurobi-threads" && i+1 < argc) grb_threads = std::stoi(argv[++i]);
    else if (a == "--sym-budget" && i+1 < argc) sym_budget = std::stod(argv[++i]);
    else if (a == "--orbit-depth" && i+1 < argc) orbit_depth = std::stoi(argv[++i]);
  }

  if (time_limit < 0.0) time_limit = 0.0;
  if (node_limit < 1) node_limit = 1;
  if (workers < 1) workers = 1;
  if (sym_budget < 0.0) sym_budget = 0.0;
  if (orbit_depth < 0) orbit_depth = 0;

  const auto t0 = std::chrono::steady_clock::now();

  {
    json out;
    out["tool"] = "orbit_branch_depth_bliss";
    out["mps"] = mps;
    out["time_limit_sec"] = time_limit;
    out["node_limit"] = node_limit;
    out["threads"] = workers;
    out["gurobi_threads"] = grb_threads;
    out["sym_budget_sec"] = sym_budget;
    out["orbit_depth"] = orbit_depth;
    out["phase"] = "started";
    out["ok"] = true;
    out["checkpoint"] = g_checkpoint.load();
    std::cout << out.dump() << "\n";
    std::cout.flush();
  }

  try {
    set_checkpoint("main:gurobi_env");
    GRBEnv env(true);
    env.set(GRB_IntParam_LogToConsole, 0);
    env.start();

    set_checkpoint("main:gurobi_model_load");
    GRBModel model(env, mps);
    model.set(GRB_IntParam_OutputFlag, 0);
    // main thread only; don't force Threads here

    int numVars = model.get(GRB_IntAttr_NumVars);
    int numConstrs = model.get(GRB_IntAttr_NumConstrs);
    int numQCons = model.get(GRB_IntAttr_NumQConstrs);

    GRBVar* orig_vars = model.getVars();
    std::vector<string> vname((size_t)numVars);
    for (int i = 0; i < numVars; ++i) {
      vname[(size_t)i] = orig_vars[i].get(GRB_StringAttr_VarName);
    }
    delete[] orig_vars;

    set_checkpoint("main:build_exact_graph");
    int numVarsCheck = 0, numBin = 0, numConstr = 0, numQ = 0;
    BuiltGraph graph = build_exact_graph_from_gurobi(model, numVarsCheck, numBin, numConstr, numQ);
    validate_BG(graph);

    set_checkpoint("main:serialize_BG_memfd");
    int bg_fd = create_bg_memfd_and_write(graph);

    std::unordered_map<unsigned, int> gadgetVarVtx2VarIndex;
    gadgetVarVtx2VarIndex.reserve(graph.bin_var_vertex_ids.size() * 2);
    for (int var_idx : graph.bin_var_indices) {
      gadgetVarVtx2VarIndex[ graph.var_vertex_of_index[(size_t)var_idx] ] = var_idx;
    }

    WorkStack WS;
    WS.st.reserve(1024);

    Node root;
    root.fix_bin.assign((size_t)graph.bin_var_indices.size(), (int8_t)-1);
    root.depth = 0;
    WS.st.push_back(std::move(root));

    std::atomic<int> explored{0};
    std::atomic<int> branched{0};
    std::atomic<int> handed_to_gurobi{0};
    std::atomic<bool> terminated{false};

    std::mutex inc_mtx;
    bool have_inc = false;
    double best_obj = std::numeric_limits<double>::infinity();
    std::vector<double> best_sol((size_t)numVars, 0.0);

    auto worker_fn = [&](int wid) {
      (void)wid;

      try {
        GRBEnv wenv(true);
        wenv.set(GRB_IntParam_LogToConsole, 0);
        wenv.start();

        GRBModel wmodel(wenv, mps);
        wmodel.set(GRB_IntParam_OutputFlag, 0);
        if (grb_threads > 0) wmodel.set(GRB_IntParam_Threads, grb_threads);

        GRBModel base_relax = wmodel.relax();
        base_relax.set(GRB_IntParam_OutputFlag, 0);
        if (grb_threads > 0) base_relax.set(GRB_IntParam_Threads, grb_threads);

        HelperSession helper;
        helper.bg_fd = bg_fd;
        if (!helper.start()) helper.disabled = true;

        while (true) {
          if (terminated.load(std::memory_order_relaxed)) break;

          if (time_limit > 0.0 && wall_sec_since(t0) >= time_limit) {
            terminated.store(true);
            break;
          }

          int e = explored.load(std::memory_order_relaxed);
          if (e >= node_limit) {
            terminated.store(true);
            break;
          }

          Node node;
          if (!WS.pop(node)) break;

          int my_expl = explored.fetch_add(1) + 1;
          if (my_expl > node_limit) {
            WS.worker_finished_unit();
            terminated.store(true);
            break;
          }

          // depth cutoff: hand off subtree to Gurobi
          if (node.depth >= orbit_depth) {
            double remaining = 0.0;
            if (time_limit > 0.0) {
              remaining = std::max(0.0, time_limit - wall_sec_since(t0));
              if (remaining <= 0.0) {
                terminated.store(true);
                WS.worker_finished_unit();
                break;
              }
            }

            bool local_have_inc;
            double local_best;
            {
              std::lock_guard<std::mutex> lk(inc_mtx);
              local_have_inc = have_inc;
              local_best = best_obj;
            }

            handed_to_gurobi.fetch_add(1);
            solve_submip_with_fixings(
              wmodel, graph, node, numVars, grb_threads,
              remaining, local_have_inc, local_best,
              inc_mtx, have_inc, best_obj, best_sol, terminated
            );

            WS.worker_finished_unit();
            continue;
          }

          // LP relaxation for branching decisions
          GRBModel relax(base_relax);
          relax.set(GRB_IntParam_OutputFlag, 0);
          if (grb_threads > 0) relax.set(GRB_IntParam_Threads, grb_threads);

          GRBVar* rvars_ptr = relax.getVars();
          std::vector<GRBVar> rvars((size_t)numVars);
          for (int i = 0; i < numVars; ++i) rvars[(size_t)i] = rvars_ptr[i];
          delete[] rvars_ptr;

          for (size_t bpos = 0; bpos < graph.bin_var_indices.size(); ++bpos) {
            int8_t f = node.fix_bin[bpos];
            if (f < 0) continue;
            int var_idx = graph.bin_var_indices[bpos];
            if (f == 0) {
              rvars[(size_t)var_idx].set(GRB_DoubleAttr_LB, 0.0);
              rvars[(size_t)var_idx].set(GRB_DoubleAttr_UB, 0.0);
            } else {
              rvars[(size_t)var_idx].set(GRB_DoubleAttr_LB, 1.0);
              rvars[(size_t)var_idx].set(GRB_DoubleAttr_UB, 1.0);
            }
          }
          relax.update();

          relax.optimize();
          int st = relax.get(GRB_IntAttr_Status);
          if (!(st == GRB_OPTIMAL || st == GRB_SUBOPTIMAL)) {
            WS.worker_finished_unit();
            continue;
          }

          double lb = relax.get(GRB_DoubleAttr_ObjVal);

          {
            std::lock_guard<std::mutex> lk(inc_mtx);
            if (have_inc && lb >= best_obj - 1e-9) {
              WS.worker_finished_unit();
              continue;
            }
          }

          std::vector<double> x((size_t)numVars, 0.0);
          for (int i = 0; i < numVars; ++i) {
            x[(size_t)i] = rvars[(size_t)i].get(GRB_DoubleAttr_X);
          }

          // check integrality on binaries
          bool integral = true;
          for (int var_idx : graph.bin_var_indices) {
            double v = x[(size_t)var_idx];
            if (std::fabs(v - std::round(v)) > 1e-6) { integral = false; break; }
          }

          // OPTION A: integral LP => accept as feasible incumbent, NO handoff here.
          if (integral) {
            std::lock_guard<std::mutex> lk(inc_mtx);
            if (!have_inc || lb < best_obj) {
              have_inc = true;
              best_obj = lb;
              best_sol = x;
            }
            WS.worker_finished_unit();
            continue;
          }

          // pick fractional seed
          int seed_bpos = pick_fractional_binary_pos(x, graph.bin_var_indices, 1e-6);
          if (seed_bpos < 0) {
            // should not happen if integral check above is correct, but be safe:
            WS.worker_finished_unit();
            continue;
          }

          int seed_var_idx = graph.bin_var_indices[(size_t)seed_bpos];
          uint32_t seed_gadget_vertex = (uint32_t)graph.var_vertex_of_index[(size_t)seed_var_idx];

          std::vector<int> orbit_var_indices;
          orbit_var_indices.push_back(seed_var_idx);

          if (!helper.disabled && sym_budget > 0.0) {
            double remaining = (time_limit > 0.0)
              ? std::max(0.0, time_limit - wall_sec_since(t0))
              : sym_budget;

            double budget = (time_limit > 0.0)
              ? std::min(sym_budget, remaining)
              : sym_budget;

            if (budget > 0.0) {
              bool orbit_term = false;
              std::vector<uint32_t> orbit_vertices;
              bool ok = session_orbit_query(helper, seed_gadget_vertex, node.fix_bin,
                                            budget, orbit_term, orbit_vertices);

              if (orbit_term) {
                terminated.store(true);
                WS.worker_finished_unit();
                break;
              }

              if (ok && !orbit_vertices.empty()) {
                std::vector<int> tmp;
                tmp.reserve(orbit_vertices.size());
                for (uint32_t gv : orbit_vertices) {
                  auto it = gadgetVarVtx2VarIndex.find((unsigned)gv);
                  if (it != gadgetVarVtx2VarIndex.end()) tmp.push_back(it->second);
                }
                std::sort(tmp.begin(), tmp.end());
                tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
                if (!tmp.empty()) orbit_var_indices = std::move(tmp);
              }
            }
          }

          // normal branching if orbit size 1
          if (orbit_var_indices.size() <= 1) {
            Node left  = node;
            Node right = node;
            left.depth  = node.depth + 1;
            right.depth = node.depth + 1;
            left.fix_bin[(size_t)seed_bpos]  = 0;
            right.fix_bin[(size_t)seed_bpos] = 1;
            WS.push(std::move(right));
            WS.push(std::move(left));
            branched.fetch_add(1);
            WS.worker_finished_unit();
            continue;
          }

          // orbit branching (chain)
          std::vector<int> orbit_bpos;
          orbit_bpos.reserve(orbit_var_indices.size());

          static thread_local std::unordered_map<int, int> var_to_bpos;
          if (var_to_bpos.empty()) {
            var_to_bpos.reserve(graph.bin_var_indices.size() * 2);
            for (int i = 0; i < (int)graph.bin_var_indices.size(); ++i) {
              var_to_bpos[graph.bin_var_indices[(size_t)i]] = i;
            }
          }

          for (int v : orbit_var_indices) {
            auto itp = var_to_bpos.find(v);
            if (itp != var_to_bpos.end()) orbit_bpos.push_back(itp->second);
          }

          std::sort(orbit_bpos.begin(), orbit_bpos.end());
          orbit_bpos.erase(std::unique(orbit_bpos.begin(), orbit_bpos.end()), orbit_bpos.end());
          if (orbit_bpos.empty()) orbit_bpos.push_back(seed_bpos);

          for (int i = (int)orbit_bpos.size() - 1; i >= 0; --i) {
            Node child = node;
            child.depth = node.depth + 1;
            for (int j = 0; j < i; ++j) child.fix_bin[(size_t)orbit_bpos[(size_t)j]] = 0;
            child.fix_bin[(size_t)orbit_bpos[(size_t)i]] = 1;
            WS.push(std::move(child));
          }

          branched.fetch_add(1);
          WS.worker_finished_unit();
        }

        helper.stop(false);
      } catch (...) {
        terminated.store(true);
        WS.stop_all();
      }
    };

    set_checkpoint("main:spawn_workers");
    std::vector<std::thread> pool;
    pool.reserve((size_t)workers);
    for (int i = 0; i < workers; ++i) pool.emplace_back(worker_fn, i);

    for (auto& th : pool) th.join();

    set_checkpoint("emit_result");
    json out;
    out["tool"] = "orbit_branch_depth_bliss";
    out["mps"] = mps;
    out["time_limit_sec"] = time_limit;
    out["node_limit"] = node_limit;
    out["threads"] = workers;
    out["gurobi_threads"] = grb_threads;
    out["sym_budget_sec"] = sym_budget;
    out["orbit_depth"] = orbit_depth;

    out["num_vars"] = numVars;
    out["num_constrs"] = numConstrs;
    out["num_qconstrs"] = numQCons;
    out["num_binary_vars"] = (int)graph.bin_var_indices.size();
    out["num_vertices"] = (int)graph.n();

    out["explored_nodes"] = explored.load();
    out["branched_nodes"] = branched.load();
    out["handoff_nodes"] = handed_to_gurobi.load();
    out["terminated_early"] = terminated.load();
    out["have_incumbent"] = have_inc;
    out["best_obj"] = have_inc ? json(best_obj) : json(nullptr);
    out["wall_runtime_sec"] = wall_sec_since(t0);
    out["ok"] = true;

    if (have_inc) {
      json bins = json::object();
      for (int var_idx : graph.bin_var_indices) {
        bins[vname[(size_t)var_idx]] = (int)llround(best_sol[(size_t)var_idx]);
      }
      out["best_binary_solution"] = bins;
    }

    std::cout << out.dump() << "\n";
    std::cout.flush();
    std::_Exit(0);
  }
  catch (GRBException& e) {
    json err;
    err["tool"] = "orbit_branch_depth_bliss";
    err["mps"] = mps;
    err["ok"] = false;
    err["error_type"] = "GRBException";
    err["error_code"] = e.getErrorCode();
    err["error_msg"] = e.getMessage();
    err["checkpoint"] = g_checkpoint.load();
    std::cout << err.dump() << "\n";
    std::cout.flush();
    std::_Exit(2);
  }
  catch (std::exception& e) {
    json err;
    err["tool"] = "orbit_branch_depth_bliss";
    err["mps"] = mps;
    err["ok"] = false;
    err["error_type"] = "std::exception";
    err["error_msg"] = e.what();
    err["checkpoint"] = g_checkpoint.load();
    std::cout << err.dump() << "\n";
    std::cout.flush();
    std::_Exit(3);
  }
}
