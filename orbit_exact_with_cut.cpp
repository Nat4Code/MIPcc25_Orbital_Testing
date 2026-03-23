// orbit_exact_with_cut.cpp
//
// Exact symmetry version (Bliss via ./orbit_helper) with SAFE helper I/O and SINGLE Gurobi solve.
//
// Adds requested diagnostics + safety:
//   - progress output during orbit computation
//   - counters: timeouts, restarts, fallback_singletons
//   - optional caps: --orbit-max-queries, --orbit-max-seconds
//
// Build:
//   g++ -O3 -DNDEBUG -march=native -flto -std=c++20 \
//     -I/opt/gurobi1301/linux64/include orbit_exact_with_cut.cpp -o orbit_exact_with_cut \
//     -L/opt/gurobi1301/linux64/lib -lgurobi_c++ -lgurobi130 \
//     -Wl,-rpath,/opt/gurobi1301/linux64/lib
//
// Run example:
//   ./orbit_exact_with_cut model.mps --symbreak --symcut --sym-budget 0.25 --verbose
//
// Flags:
//   --time-limit T           Gurobi time limit seconds (default 0 => no limit)
//   --sym-budget S           orbit query time budget seconds per seed (default 0.25)
//   --io-timeout-mult M      read timeout = (sym_budget + 0.25) * M seconds (default 4.0)
//   --helper-stderr PATH     file to write helper stderr (default orbit_helper_stderr.log)
//   --verbose                print progress banners and enable Gurobi console output
//   --symbreak               add orbit ordering constraints x1 >= x2 >= ...
//   --symcut                 add lazy no-good cuts at MIPSOL (exact orbits => safe)
//   --symcut-max-swaps K     swaps per orbit per incumbent (default 2)
//   --symcut-max-cuts  C     max lazy cuts per incumbent (default 50)
//   --orbit-progress-every N print orbit progress every N seeds (default 10)
//   --orbit-max-queries Q    cap number of orbit queries (default 1000000000)
//   --orbit-max-seconds S    cap total seconds spent in orbit queries (default 0 => no cap)

#include <gurobi_c++.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <memory>
#include <poll.h>
#include <spawn.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

extern char **environ;

using std::string;

// --------------------------- timing ---------------------------
static double now_sec() {
  using namespace std::chrono;
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static bool is_nan(double x) { return std::isnan(x); }

// --------------------------- CLI helpers ---------------------------
static bool has_flag(int argc, char** argv, const string& flag) {
  for (int i = 2; i < argc; ++i) if (string(argv[i]) == flag) return true;
  return false;
}
static string parse_flag_str(int argc, char** argv, const string& flag, const string& defval) {
  for (int i = 2; i + 1 < argc; ++i) if (string(argv[i]) == flag) return string(argv[i+1]);
  return defval;
}
static double parse_flag_double(int argc, char** argv, const string& flag, double defval) {
  for (int i = 2; i + 1 < argc; ++i) {
    if (string(argv[i]) == flag) {
      try { return std::stod(argv[i+1]); } catch (...) { return defval; }
    }
  }
  return defval;
}
static int parse_flag_int(int argc, char** argv, const string& flag, int defval) {
  for (int i = 2; i + 1 < argc; ++i) {
    if (string(argv[i]) == flag) {
      try { return std::stoi(argv[i+1]); } catch (...) { return defval; }
    }
  }
  return defval;
}
static long long parse_flag_ll(int argc, char** argv, const string& flag, long long defval) {
  for (int i = 2; i + 1 < argc; ++i) {
    if (string(argv[i]) == flag) {
      try { return std::stoll(argv[i+1]); } catch (...) { return defval; }
    }
  }
  return defval;
}

// --------------------------- fd I/O helpers (timeout-safe) ---------------------------
static bool write_all_fd(int fd, const void* buf, size_t n) {
  const char* p = (const char*)buf;
  size_t off = 0;
  while (off < n) {
    ssize_t w = ::write(fd, p + off, n - off);
    if (w > 0) { off += (size_t)w; continue; }
    if (w < 0 && errno == EINTR) continue;
    return false;
  }
  return true;
}

static bool wait_fd(int fd, short events, int timeout_ms) {
  struct pollfd pfd;
  pfd.fd = fd;
  pfd.events = events;
  while (true) {
    int rc = ::poll(&pfd, 1, timeout_ms);
    if (rc > 0) {
      if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) return false;
      return (pfd.revents & events) != 0;
    }
    if (rc == 0) return false;   // timeout
    if (errno == EINTR) continue;
    return false;
  }
}

static bool read_all_fd_timeout(int fd, void* buf, size_t n, int timeout_ms) {
  char* p = (char*)buf;
  size_t off = 0;
  while (off < n) {
    if (!wait_fd(fd, POLLIN, timeout_ms)) return false;
    ssize_t r = ::read(fd, p + off, n - off);
    if (r > 0) { off += (size_t)r; continue; }
    if (r < 0 && errno == EINTR) continue;
    return false; // EOF or error
  }
  return true;
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

// --------------------------- build exact graph from gurobi ---------------------------
static BuiltGraph build_exact_graph_from_gurobi(GRBModel& model,
                                               int& numVarsOut,
                                               int& numBinOut,
                                               int& numConstrOut,
                                               int& numQConsOut) {
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
  std::unordered_map<std::string, unsigned> name2varvid;
  name2varvid.reserve((size_t)numVars * 2);
  for (int var_idx = 0; var_idx < numVars; ++var_idx) {
    name2varvid.emplace(vars[var_idx].get(GRB_StringAttr_VarName),
                        graph.var_vertex_of_index[(size_t)var_idx]);
  }

  // linear constraints
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
  std::vector<unsigned> qcon_vertex_id((size_t)numQCons, 0);
  for (int qi = 0; qi < numQCons; ++qi) {
    double rhs = qcons[qi].get(GRB_DoubleAttr_QCRHS);
    char sense = qcons[qi].get(GRB_CharAttr_QCSense);

    std::string qkey_s = "Q|";
    qkey_s.push_back(sense);
    qkey_s += "|rhs:" + std::to_string(qkey(rhs));
    qcon_vertex_id[(size_t)qi] = graph.add_vertex(colors.get(qkey_s));
  }

  // quadratic content
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

  graph.dedup_and_build_edgelist();
  return graph;
}

// -------------------- memfd graph serialization for helper --------------------
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
  std::string stderr_path;

  void stop(bool kill_hard = true) {
    if (wfd >= 0) { close(wfd); wfd = -1; }
    if (rfd >= 0) { close(rfd); rfd = -1; }
    if (pid > 0) {
      int st = 0;
      if (kill_hard) kill(pid, SIGKILL);
      waitpid(pid, &st, 0);
      pid = -1;
    }
  }

  bool start() {
    stop(true);

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

    int errfd = ::open(stderr_path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (errfd >= 0) {
      posix_spawn_file_actions_adddup2(&fa, errfd, STDERR_FILENO);
      posix_spawn_file_actions_addclose(&fa, errfd);
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
      return false;
    }

    wfd = inpipe[1];
    rfd = outpipe[0];
    return true;
  }
};

// -------------------- orbit query (safe) --------------------
static bool session_orbit_query_once(HelperSession& session,
                                    uint32_t seed_vertex,
                                    const std::vector<int8_t>& fix_by_bin,
                                    double time_limit_sec,
                                    int read_timeout_ms,
                                    bool& terminated,
                                    std::vector<uint32_t>& orbit_vertices_out) {
  terminated = false;
  orbit_vertices_out.clear();

  uint8_t req_type = 0;
  if (!write_all_fd(session.wfd, &req_type, sizeof(req_type))) return false;
  if (!write_all_fd(session.wfd, &seed_vertex, sizeof(seed_vertex))) return false;
  if (!write_all_fd(session.wfd, &time_limit_sec, sizeof(time_limit_sec))) return false;
  if (!fix_by_bin.empty() &&
      !write_all_fd(session.wfd, fix_by_bin.data(), fix_by_bin.size() * sizeof(int8_t))) return false;

  uint8_t ok = 0, term = 0;
  if (!read_all_fd_timeout(session.rfd, &ok, sizeof(ok), read_timeout_ms)) return false;
  if (!read_all_fd_timeout(session.rfd, &term, sizeof(term), read_timeout_ms)) return false;

  terminated = (term == 1);
  if (ok != 1) return false;

  uint32_t k = 0;
  if (!read_all_fd_timeout(session.rfd, &k, sizeof(k), read_timeout_ms)) return false;
  orbit_vertices_out.resize(k);
  if (k > 0 && !read_all_fd_timeout(session.rfd, orbit_vertices_out.data(), (size_t)k * sizeof(uint32_t), read_timeout_ms))
    return false;

  return true;
}

static bool session_orbit_query_safe(HelperSession& session,
                                    uint32_t seed_vertex,
                                    const std::vector<int8_t>& fix_by_bin,
                                    double time_limit_sec,
                                    int read_timeout_ms,
                                    bool& terminated,
                                    std::vector<uint32_t>& orbit_vertices_out,
                                    long long& restarts_out) {
  if (session_orbit_query_once(session, seed_vertex, fix_by_bin, time_limit_sec, read_timeout_ms,
                              terminated, orbit_vertices_out)) return true;

  // restart and retry once
  restarts_out++;
  session.stop(true);
  if (!session.start()) return false;

  terminated = false;
  orbit_vertices_out.clear();
  return session_orbit_query_once(session, seed_vertex, fix_by_bin, time_limit_sec, read_timeout_ms,
                                 terminated, orbit_vertices_out);
}

// ----------------------------- exact-orbit symmetry cuts callback -----------------------------
struct SymNoGoodCallback : public GRBCallback {
  std::vector<GRBVar> vars;
  std::vector<char> vtype;
  std::vector<std::vector<int>> orbits;

  bool verbose = false;
  int max_swaps_per_orbit = 2;
  int max_cuts_per_inc = 50;

  size_t last_hash = 0;

  static size_t hash_binary_assignment(const std::vector<double>& x) {
    size_t h = 1469598103934665603ull;
    for (double xi : x) {
      unsigned char b = (xi > 0.5) ? 1 : 0;
      h ^= (size_t)b;
      h *= 1099511628211ull;
    }
    return h;
  }

  void callback() override {
    try {
      if (where != GRB_CB_MIPSOL) return;

      const int n = (int)vars.size();
      std::vector<double> sol(n, 0.0);
      for (int i = 0; i < n; ++i) sol[i] = getSolution(vars[i]);

      size_t h = hash_binary_assignment(sol);
      if (h == last_hash) return;
      last_hash = h;

      int cuts_added = 0;

      for (const auto& O : orbits) {
        if (cuts_added >= max_cuts_per_inc) break;
        if ((int)O.size() < 2) continue;

        bool all_bin = true;
        for (int idx : O) {
          if (idx < 0 || idx >= n) { all_bin = false; break; }
          if (vtype[idx] != GRB_BINARY) { all_bin = false; break; }
        }
        if (!all_bin) continue;

        int i0 = O[0];
        int lim = std::min((int)O.size() - 1, max_swaps_per_orbit);

        for (int t = 1; t <= lim && cuts_added < max_cuts_per_inc; ++t) {
          int j = O[t];

          double yi0 = sol[j];
          double yj  = sol[i0];

          bool diff = ((sol[i0] > 0.5) != (yi0 > 0.5)) || ((sol[j] > 0.5) != (yj > 0.5));
          if (!diff) continue;

          GRBLinExpr expr = 0.0;
          for (int idx : O) {
            bool y1;
            if (idx == i0) y1 = (yi0 > 0.5);
            else if (idx == j) y1 = (yj > 0.5);
            else y1 = (sol[idx] > 0.5);

            if (y1) expr += (1.0 - vars[idx]);
            else    expr += vars[idx];
          }

          addLazy(expr >= 1.0);
          ++cuts_added;
        }
      }

      if (verbose && cuts_added > 0) {
        std::cout << "[symcut] added " << cuts_added << " lazy cuts at MIPSOL\n";
        std::cout.flush();
      }
    } catch (...) {}
  }
};

// ----------------------------- main -----------------------------
int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: orbit_exact_with_cut model.mps [--time-limit T] [--sym-budget S] [--verbose] [--symbreak] [--symcut]\n";
    return 1;
  }

  const std::string mps = argv[1];

  const double TIME_LIMIT = parse_flag_double(argc, argv, "--time-limit", 0.0);
  const double SYM_BUDGET = parse_flag_double(argc, argv, "--sym-budget", 0.25);
  const double IO_TIMEOUT_MULT = parse_flag_double(argc, argv, "--io-timeout-mult", 4.0);
  const double ORBIT_MAX_SECONDS = parse_flag_double(argc, argv, "--orbit-max-seconds", 0.0);
  const long long ORBIT_MAX_QUERIES = parse_flag_ll(argc, argv, "--orbit-max-queries", (long long)1000000000);

  const int PROG_EVERY = parse_flag_int(argc, argv, "--orbit-progress-every", 10);

  const bool VERBOSE = has_flag(argc, argv, "--verbose");
  const bool SYMBREAK = has_flag(argc, argv, "--symbreak");
  const bool SYMCUT   = has_flag(argc, argv, "--symcut");
  const int  SYMCUT_MAX_SWAPS = parse_flag_int(argc, argv, "--symcut-max-swaps", 2);
  const int  SYMCUT_MAX_CUTS  = parse_flag_int(argc, argv, "--symcut-max-cuts", 50);

  const std::string HELPER_STDERR = parse_flag_str(argc, argv, "--helper-stderr", "orbit_helper_stderr.log");

  int read_timeout_ms = (int)std::llround(1000.0 * std::max(0.25, (SYM_BUDGET + 0.25) * IO_TIMEOUT_MULT));
  if (read_timeout_ms < 200) read_timeout_ms = 200;

  double t0 = now_sec();
  double t_graph=0, t_orbits=0, t_relax=0, t_score=0, t_solve=0;

  // orbit diagnostics
  long long orbit_queries = 0;
  long long orbit_restarts = 0;
  long long orbit_timeouts_or_fail = 0;
  long long fallback_singletons = 0;
  long long covered_bins = 0;

  try {
    GRBEnv env(true);
    env.set(GRB_IntParam_LogToConsole, VERBOSE ? 1 : 0);
    env.start();

    GRBModel model(env, mps);
    model.set(GRB_IntParam_OutputFlag, VERBOSE ? 1 : 0);

    const int numVars = model.get(GRB_IntAttr_NumVars);
    const int numConstr = model.get(GRB_IntAttr_NumConstrs);
    const int numQConstr = model.get(GRB_IntAttr_NumQConstrs);

    if (VERBOSE) {
      std::cout << "[phase] loaded model: vars=" << numVars
                << " constr=" << numConstr
                << " qconstr=" << numQConstr << "\n";
      std::cout.flush();
    }

    if (numConstr == 0 && numQConstr == 0) {
      if (VERBOSE) {
        std::cout << "[info] 0 constraints; skipping orbit symmetry & folding.\n";
        std::cout.flush();
      }
      GRBModel full(model);
      full.set(GRB_IntParam_OutputFlag, VERBOSE ? 1 : 0);
      if (TIME_LIMIT > 0.0) full.set(GRB_DoubleParam_TimeLimit, TIME_LIMIT);
      full.optimize();

      int st = full.get(GRB_IntAttr_Status);
      double best_obj = NAN;
      if (full.get(GRB_IntAttr_SolCount) > 0) best_obj = full.get(GRB_DoubleAttr_ObjVal);

      std::ostringstream ss;
      ss << "{";
      ss << "\"tool\":\"orbit_exact_with_cut_cpp\",";
      ss << "\"mps\":\"" << mps << "\",";
      ss << "\"degenerate_no_constraints\":true,";
      ss << "\"status\":" << st << ",";
      ss << "\"best_obj\":" << (is_nan(best_obj) ? string("null") : std::to_string(best_obj)) << ",";
      ss << "\"runtime_sec\":" << full.get(GRB_DoubleAttr_Runtime) << ",";
      ss << "\"wall_runtime_sec\":" << (now_sec() - t0);
      ss << "}";
      std::cout << ss.str() << "\n";
      return 0;
    }

    // cache vtypes + obj
    GRBVar* vars_ptr = (numVars > 0) ? model.getVars() : nullptr;
    std::vector<char>   vtype((size_t)numVars);
    std::vector<double> obj((size_t)numVars);
    for (int i = 0; i < numVars; ++i) {
      vtype[(size_t)i] = vars_ptr[i].get(GRB_CharAttr_VType);
      obj[(size_t)i]   = vars_ptr[i].get(GRB_DoubleAttr_Obj);
    }
    if (vars_ptr) delete[] vars_ptr;

    // Build exact gadget graph
    if (VERBOSE) { std::cout << "[phase] building exact graph\n"; std::cout.flush(); }
    double tsec = now_sec();
    int numVarsCheck=0, numBin=0, numConstrOut=0, numQOut=0;
    BuiltGraph BG = build_exact_graph_from_gurobi(model, numVarsCheck, numBin, numConstrOut, numQOut);
    t_graph += now_sec() - tsec;

    if (VERBOSE) {
      std::cout << "[phase] graph built: vertices=" << BG.n()
                << " bin=" << BG.bin_var_indices.size()
                << " edges=" << BG.edges.size() << "\n";
      std::cout.flush();
    }

    // gadget vertex -> var index for binary vars
    std::unordered_map<unsigned,int> gadgetVarVtx2VarIndex;
    gadgetVarVtx2VarIndex.reserve(BG.bin_var_vertex_ids.size() * 2);
    for (int var_idx : BG.bin_var_indices) {
      gadgetVarVtx2VarIndex[ BG.var_vertex_of_index[(size_t)var_idx] ] = var_idx;
    }

    // Start helper
    if (VERBOSE) {
      std::cout << "[phase] starting orbit_helper (stderr -> " << HELPER_STDERR << ")\n";
      std::cout.flush();
    }
    int bg_fd = create_bg_memfd_and_write(BG);
    HelperSession helper;
    helper.bg_fd = bg_fd;
    helper.stderr_path = HELPER_STDERR;

    if (!helper.start()) {
      close(bg_fd);
      throw std::runtime_error("Failed to start ./orbit_helper (required). See helper stderr log.");
    }

    // Compute orbits
    if (VERBOSE) {
      std::cout << "[phase] computing exact orbits (sym_budget=" << SYM_BUDGET
                << "s, read_timeout_ms=" << read_timeout_ms
                << ", max_queries=" << ORBIT_MAX_QUERIES
                << ", max_seconds=" << ORBIT_MAX_SECONDS << ")\n";
      std::cout.flush();
    }

    tsec = now_sec();
    double orbit_start = now_sec();

    std::vector<std::vector<int>> orbits;
    orbits.reserve((size_t)BG.bin_var_indices.size());

    std::unordered_map<int,int> var_to_bpos;
    var_to_bpos.reserve(BG.bin_var_indices.size() * 2);
    for (int bpos = 0; bpos < (int)BG.bin_var_indices.size(); ++bpos) {
      var_to_bpos[ BG.bin_var_indices[(size_t)bpos] ] = bpos;
    }

    std::vector<int8_t> fix_by_bin((size_t)BG.bin_var_indices.size(), (int8_t)-1);
    std::vector<char> seen((size_t)BG.bin_var_indices.size(), 0);

    for (int bpos = 0; bpos < (int)BG.bin_var_indices.size(); ++bpos) {
      if (seen[(size_t)bpos]) continue;

      if (orbit_queries >= ORBIT_MAX_QUERIES) break;
      if (ORBIT_MAX_SECONDS > 0.0 && (now_sec() - orbit_start) >= ORBIT_MAX_SECONDS) break;

      uint32_t seed_gv = (uint32_t)BG.bin_var_vertex_ids[(size_t)bpos];

      bool terminated = false;
      std::vector<uint32_t> orbit_vertices;

      orbit_queries++;
      bool ok = session_orbit_query_safe(helper, seed_gv, fix_by_bin, SYM_BUDGET, read_timeout_ms,
                                         terminated, orbit_vertices, orbit_restarts);

      bool used_singleton = false;
      if (!ok || terminated || orbit_vertices.empty()) {
        orbit_timeouts_or_fail++;
        used_singleton = true;
        orbit_vertices.clear();
        orbit_vertices.push_back(seed_gv);
      }

      std::vector<int> orbit_vars;
      orbit_vars.reserve(orbit_vertices.size());
      for (uint32_t gv : orbit_vertices) {
        auto it = gadgetVarVtx2VarIndex.find((unsigned)gv);
        if (it != gadgetVarVtx2VarIndex.end()) orbit_vars.push_back(it->second);
      }

      std::sort(orbit_vars.begin(), orbit_vars.end());
      orbit_vars.erase(std::unique(orbit_vars.begin(), orbit_vars.end()), orbit_vars.end());
      if (orbit_vars.empty()) {
        used_singleton = true;
        orbit_vars.push_back(BG.bin_var_indices[(size_t)bpos]);
      }

      // mark covered
      for (int var_idx : orbit_vars) {
        auto itp = var_to_bpos.find(var_idx);
        if (itp != var_to_bpos.end() && !seen[(size_t)itp->second]) {
          seen[(size_t)itp->second] = 1;
          covered_bins++;
        }
      }

      if (used_singleton) fallback_singletons++;

      orbits.push_back(std::move(orbit_vars));

      if (VERBOSE && PROG_EVERY > 0 && (orbit_queries % PROG_EVERY == 0)) {
        double elapsed = now_sec() - orbit_start;
        std::cout << "[orbits] queries=" << orbit_queries
                  << " orbits=" << orbits.size()
                  << " covered=" << covered_bins << "/" << BG.bin_var_indices.size()
                  << " restarts=" << orbit_restarts
                  << " fail=" << orbit_timeouts_or_fail
                  << " singletons=" << fallback_singletons
                  << " elapsed=" << elapsed << "s\n";
        std::cout.flush();
      }
    }

    // For any remaining uncovered bins, add singleton orbits (explicit fallback)
    for (int bpos = 0; bpos < (int)BG.bin_var_indices.size(); ++bpos) {
      if (!seen[(size_t)bpos]) {
        fallback_singletons++;
        covered_bins++;
        orbits.push_back(std::vector<int>{ BG.bin_var_indices[(size_t)bpos] });
        seen[(size_t)bpos] = 1;
      }
    }

    std::sort(orbits.begin(), orbits.end(),
              [](const auto& a, const auto& b){ return a[0] < b[0]; });

    t_orbits += now_sec() - tsec;

    if (VERBOSE) {
      double elapsed = now_sec() - orbit_start;
      size_t big = 0;
      for (auto& O : orbits) if (O.size() >= 10) ++big;
      std::cout << "[phase] orbits done: orbits=" << orbits.size()
                << " covered=" << covered_bins << "/" << BG.bin_var_indices.size()
                << " restarts=" << orbit_restarts
                << " fail=" << orbit_timeouts_or_fail
                << " singletons=" << fallback_singletons
                << " big(>=10)=" << big
                << " elapsed=" << elapsed << "s\n";
      std::cout.flush();
    }

    // Folded relaxation
    if (VERBOSE) { std::cout << "[phase] solving folded relaxation\n"; std::cout.flush(); }
    tsec = now_sec();
    GRBModel relax = model.relax();
    relax.set(GRB_IntParam_OutputFlag, 0);

    for (const auto& O : orbits) {
      if (O.size() <= 1) continue;
      GRBVar v0 = relax.getVar(O[0]);
      for (size_t k = 1; k < O.size(); ++k) {
        relax.addConstr(v0 - relax.getVar(O[k]) == 0);
      }
    }

    relax.update();
    relax.optimize();
    t_relax += now_sec() - tsec;

    std::vector<double> x((size_t)numVars, 0.0);
    for (int i = 0; i < numVars; ++i) {
      try { x[(size_t)i] = relax.getVar(i).get(GRB_DoubleAttr_X); }
      catch (...) { x[(size_t)i] = 0.0; }
    }

    // Score + rounding
    if (VERBOSE) { std::cout << "[phase] building MIP start\n"; std::cout.flush(); }
    tsec = now_sec();
    std::unordered_set<int> fix1;
    fix1.reserve((size_t)BG.bin_var_indices.size() * 2);

    std::vector<std::pair<double,int>> scores;
    scores.reserve(orbits.size());
    for (size_t oi = 0; oi < orbits.size(); ++oi) {
      const auto& O = orbits[oi];
      if (O.size() <= 1) continue;
      double s = 0.0;
      for (int idx : O) s += obj[(size_t)idx] * x[(size_t)idx];
      scores.emplace_back(s, (int)oi);
    }
    std::sort(scores.begin(), scores.end(), std::greater<>());

    double threshold = 0.0;
    if (!scores.empty()) threshold = scores[0].first * 0.1;
    for (auto &sc : scores) {
      if (sc.first < threshold) break;
      const auto& O = orbits[(size_t)sc.second];
      for (int idx : O) fix1.insert(idx);
    }

    std::vector<std::pair<double,int>> frac;
    frac.reserve((size_t)BG.bin_var_indices.size());
    for (int var_idx : BG.bin_var_indices) {
      if (fix1.count(var_idx)) continue;
      double xi = x[(size_t)var_idx];
      if (xi > 1e-6 && xi < 0.9999) frac.emplace_back(obj[(size_t)var_idx] * (xi - 0.5), var_idx);
    }
    std::sort(frac.begin(), frac.end(), std::greater<>());
    int max_rounds = (int)frac.size()/3 + 2;
    for (int i = 0; i < (int)frac.size() && i < max_rounds; ++i) fix1.insert(frac[i].second);

    std::vector<double> x2 = x;
    for (int idx : fix1) x2[(size_t)idx] = 1.0;
    for (int var_idx : BG.bin_var_indices) {
      if (!fix1.count(var_idx)) x2[(size_t)var_idx] = (x2[(size_t)var_idx] > 0.5 ? 1.0 : 0.0);
    }
    t_score += now_sec() - tsec;

    // Full solve
    if (VERBOSE) { std::cout << "[phase] solving full MIP\n"; std::cout.flush(); }

    GRBModel full(model);
    full.set(GRB_IntParam_OutputFlag, VERBOSE ? 1 : 0);
    if (TIME_LIMIT > 0.0) full.set(GRB_DoubleParam_TimeLimit, TIME_LIMIT);

    for (int i = 0; i < numVars; ++i) full.getVar(i).set(GRB_DoubleAttr_Start, x2[(size_t)i]);

    if (SYMBREAK) {
      int added = 0;
      for (auto O : orbits) {
        if (O.size() <= 1) continue;
        std::sort(O.begin(), O.end());
        for (size_t k = 0; k + 1 < O.size(); ++k) {
          full.addConstr(full.getVar(O[k]) >= full.getVar(O[k+1]));
          ++added;
        }
      }
      if (VERBOSE) {
        std::cout << "[symbreak] added " << added << " orbit-order constraints\n";
        std::cout.flush();
      }
    }

    std::unique_ptr<SymNoGoodCallback> cb;
    if (SYMCUT) {
      full.set(GRB_IntParam_LazyConstraints, 1);

      cb.reset(new SymNoGoodCallback());
      cb->verbose = VERBOSE;
      cb->max_swaps_per_orbit = std::max(0, SYMCUT_MAX_SWAPS);
      cb->max_cuts_per_inc = std::max(1, SYMCUT_MAX_CUTS);
      cb->orbits = orbits;

      cb->vars.reserve((size_t)numVars);
      cb->vtype.reserve((size_t)numVars);
      for (int i = 0; i < numVars; ++i) {
        cb->vars.push_back(full.getVar(i));
        cb->vtype.push_back(vtype[(size_t)i]);
      }
      full.setCallback(cb.get());

      if (VERBOSE) {
        std::cout << "[symcut] enabled: max_swaps=" << cb->max_swaps_per_orbit
                  << " max_cuts=" << cb->max_cuts_per_inc << "\n";
        std::cout.flush();
      }
    } else {
      full.setCallback(nullptr);
    }

    full.update();

    tsec = now_sec();
    full.optimize();
    t_solve += now_sec() - tsec;

    int status = full.get(GRB_IntAttr_Status);
    double best_obj = NAN;
    if (full.get(GRB_IntAttr_SolCount) > 0) best_obj = full.get(GRB_DoubleAttr_ObjVal);

    helper.stop(true);
    close(bg_fd);

    double wall = now_sec() - t0;

    std::ostringstream ss;
    ss << "{";
    ss << "\"tool\":\"orbit_exact_with_cut_cpp\",";
    ss << "\"mps\":\"" << mps << "\",";
    ss << "\"verbose\":" << (VERBOSE ? "true" : "false") << ",";
    ss << "\"time_limit_sec\":" << TIME_LIMIT << ",";
    ss << "\"sym_budget_sec\":" << SYM_BUDGET << ",";
    ss << "\"read_timeout_ms\":" << read_timeout_ms << ",";
    ss << "\"helper_stderr\":\"" << HELPER_STDERR << "\",";
    ss << "\"symbreak\":" << (SYMBREAK ? "true" : "false") << ",";
    ss << "\"symcut\":" << (SYMCUT ? "true" : "false") << ",";
    ss << "\"orbit_queries\":" << orbit_queries << ",";
    ss << "\"orbit_restarts\":" << orbit_restarts << ",";
    ss << "\"orbit_fail\":" << orbit_timeouts_or_fail << ",";
    ss << "\"fallback_singletons\":" << fallback_singletons << ",";
    ss << "\"num_vars\":" << numVars << ",";
    ss << "\"num_bin\":" << (int)BG.bin_var_indices.size() << ",";
    ss << "\"num_orbits\":" << (int)orbits.size() << ",";
    ss << "\"status\":" << status << ",";
    ss << "\"best_obj\":" << (is_nan(best_obj) ? string("null") : std::to_string(best_obj)) << ",";
    ss << "\"runtime_sec\":" << full.get(GRB_DoubleAttr_Runtime) << ",";
    ss << "\"t_build_graph\":" << t_graph << ",";
    ss << "\"t_orbits\":" << t_orbits << ",";
    ss << "\"t_relax\":" << t_relax << ",";
    ss << "\"t_score\":" << t_score << ",";
    ss << "\"t_solve\":" << t_solve << ",";
    ss << "\"wall_runtime_sec\":" << wall;
    ss << "}";

    std::cout << ss.str() << "\n";
    std::cout.flush();
    return 0;
  }
  catch (GRBException& e) {
    std::cerr << "Gurobi error (" << e.getErrorCode() << "): " << e.getMessage() << "\n";
    return 2;
  }
  catch (std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 3;
  }
}