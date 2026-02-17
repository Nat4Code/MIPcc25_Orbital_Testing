/////////////////////////////////////////////////////////////////////////////////
// orbit_helper.cpp
//
// Persistent helper for orbit queries using bliss.
//
// Reads a serialized "BuiltGraph" from --bg-fd FD.
// Handles requests on stdin; writes responses to stdout.
//
// Protocol:
// req_type (uint8)
// 0: orbit query
//   seed(uint32) tlim(double) fix[numBin](int8)
//   resp: ok(uint8) term(uint8) k(uint32) orbit[k](uint32)
//
// 1: base reps (kept for compatibility; parent now skips this)
//   tlim(double) fix[numBin](int8)
//   resp: ok(uint8) term(uint8) nb(uint32) reps[nb](uint32)
/////////////////////////////////////////////////////////////////////////////////

#include <bliss/graph.hh>
#include <bliss/stats.hh>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <unistd.h>

using std::size_t;

// --------------------------- robust I/O helpers ---------------------------
// read exactly n bytes from fd into buf. Return false on EOF/error...
static bool read_all(int fd, void* buf, size_t n) {
  char* out = (char*)buf;
  size_t offset = 0;
  while (offset < n) {
    ssize_t r = ::read(fd, out + offset, n - offset);
    if (r <= 0) return false;
    offset += (size_t)r;
  }
  return true;
}

// write exactly n bytes from buf to fd. return false on error...
static bool write_all(int fd, const void* buf, size_t n) {
  const char* in = (const char*)buf;
  size_t offset = 0;
  while (offset < n) {
    ssize_t w = ::write(fd, in + offset, n - offset);
    if (w <= 0) return false;
    offset += (size_t)w;
  }
  return true;
}

// --------------------------- DSU / Union-Find ----------------------------------------
// classic disjoint-set union structure: used to accumulate orbits from automorphisms...
struct DSU {
  std::vector<int> parent;
  std::vector<int> rank;

  explicit DSU(int n = 0) : parent(n), rank(n, 0) {
    for (int i = 0; i < n; ++i) parent[i] = i;
  }

  int find(int a) {
    return parent[a] == a ? a : parent[a] = find(parent[a]);
  }

  void unite(int a, int b) {
    a = find(a);
    b = find(b);
    if (a == b) return;
    if (rank[a] < rank[b]) std::swap(a, b);
    parent[b] = a;
    if (rank[a] == rank[b]) rank[a]++;
  }
};

// --------------------------- serialized graph metadata ---------------------------
// this is the graph payload passed from the parent process via memfd.
// it contains:
//  - number of vertices N
//  - number of "binary gadget vertices" numBin
//  - a base vertex color for each vertex
//  - a list of vertex ids corresponding to binary variables (bin_vertex_ids)
//  - an undirected edge list (u,v) pairs
struct GraphData {
  uint32_t num_vertices = 0;            // N
  uint32_t num_binary_vertices = 0;     // numBin

  std::vector<uint32_t> base_color;     // size N
  std::vector<uint32_t> bin_vertex_ids; // size numBin
  std::vector<std::pair<uint32_t, uint32_t>> edges;

  // convenience mask: is this vertex one of the binary gadget vertices?
  std::vector<char> is_binary_vertex;   // size N (0/1)
};

// read the exact binary format written by the parent (create_bg_memfd_and_write).
static GraphData load_graph_from_fd(int bg_fd) {
  GraphData graph;

  uint32_t magic = 0, ver = 0;
  if (!read_all(bg_fd, &magic, sizeof(magic))) throw std::runtime_error("read magic failed");
  if (!read_all(bg_fd, &ver, sizeof(ver))) throw std::runtime_error("read ver failed");
  if (magic != 0x4f424247u || ver != 1u) throw std::runtime_error("bad bg format");

  if (!read_all(bg_fd, &graph.num_vertices, sizeof(graph.num_vertices))) throw std::runtime_error("read N failed");
  if (!read_all(bg_fd, &graph.num_binary_vertices, sizeof(graph.num_binary_vertices))) throw std::runtime_error("read numBin failed");

  graph.base_color.resize(graph.num_vertices);
  if (graph.num_vertices > 0 &&
      !read_all(bg_fd, graph.base_color.data(), (size_t)graph.num_vertices * sizeof(uint32_t))) {
    throw std::runtime_error("read base_color failed");
  }

  graph.bin_vertex_ids.resize(graph.num_binary_vertices);
  if (graph.num_binary_vertices > 0 &&
      !read_all(bg_fd, graph.bin_vertex_ids.data(), (size_t)graph.num_binary_vertices * sizeof(uint32_t))) {
    throw std::runtime_error("read bin_vid failed");
  }

  uint32_t m = 0;
  if (!read_all(bg_fd, &m, sizeof(m))) throw std::runtime_error("read m failed");
  graph.edges.resize(m);
  for (uint32_t i = 0; i < m; ++i) {
    uint32_t u = 0, v = 0;
    if (!read_all(bg_fd, &u, sizeof(u))) throw std::runtime_error("read edge u failed");
    if (!read_all(bg_fd, &v, sizeof(v))) throw std::runtime_error("read edge v failed");
    graph.edges[i] = {u, v};
  }

  // Build the quick membership array for "is this a binary gadget vertex?"
  graph.is_binary_vertex.assign(graph.num_vertices, 0);
  for (uint32_t i = 0; i < graph.num_binary_vertices; ++i) {
    uint32_t vid = graph.bin_vertex_ids[i];
    if (vid < graph.num_vertices) graph.is_binary_vertex[vid] = 1;
  }

  return graph;
}

// --------------------------- orbit query core ---------------------------
// given:
//  - a fixed "base graph" (colors + edges)
//  - a seed vertex (must be a binary gadget vertex to be meaningful)
//  - fixings for each binary gadget vertex in the helper’s bin-order:
//      -1 = unfixed, 0 = fixed-to-0, 1 = fixed-to-1
//  - a time limit in seconds (<=0 means no time limit)
//
// this function:
//  1) Builds a bliss graph with colors modified by fixing-buckets
//  2) Runs bliss automorphism search (with optional termination callback)
//  3) Unions binary vertices that map to each other in any generator automorphism
//  4) Returns the orbit of the seed restricted to binary gadget vertices
static bool build_and_solve_orbit(const GraphData& graph,
                                 uint32_t seed_vertex,
                                 const std::vector<int8_t>& fixing_by_binpos,
                                 double time_limit_sec,
                                 bool& terminated,
                                 std::vector<uint32_t>& orbit_vertices_out) {
  terminated = false;
  orbit_vertices_out.clear();

  // Defensive fallbacks: if anything is off, return trivial orbit = {seed}.
  if (graph.num_vertices == 0 || graph.num_binary_vertices == 0) {
    orbit_vertices_out.push_back(seed_vertex);
    return true;
  }
  if (seed_vertex >= graph.num_vertices || !graph.is_binary_vertex[seed_vertex]) {
    orbit_vertices_out.push_back(seed_vertex);
    return true;
  }
  if (fixing_by_binpos.size() != graph.num_binary_vertices) {
    orbit_vertices_out.push_back(seed_vertex);
    return true;
  }

  // step 1: compute vertex colors incorporating fixings on binary gadget vertices.
  // the parent provides base_color; here we refine each binary vertex color by putting it
  // into a "bucket" representing fixed-to-0 / fixed-to-1 / unfixed.
  //
  // bucket:
  //   0 => unfixed (-1)
  //   1 => fixed 0
  //   2 => fixed 1
  //
  // new_color = base_color * 4 + bucket
  std::vector<uint32_t> refined_color = graph.base_color;
  for (uint32_t bin_pos = 0; bin_pos < graph.num_binary_vertices; ++bin_pos) {
    uint32_t vtx = graph.bin_vertex_ids[bin_pos];
    if (vtx >= graph.num_vertices) continue;

    int8_t fix_state = fixing_by_binpos[bin_pos];
    uint32_t bucket = 0;
    if (fix_state == 0) bucket = 1;
    else if (fix_state == 1) bucket = 2;

    refined_color[vtx] = refined_color[vtx] * 4u + bucket;
  }

  // step 2: build bliss graph.
  // bliss::Graph expects:
  //  - add_vertex(color) for each vertex
  //  - add_edge(u,v) for each undirected edge
  bliss::Graph bliss_graph;

  for (uint32_t i = 0; i < graph.num_vertices; ++i) {
    bliss_graph.add_vertex((unsigned)refined_color[i]);
  }
  for (const auto& e : graph.edges) {
    if (e.first < graph.num_vertices && e.second < graph.num_vertices && e.first != e.second) {
      bliss_graph.add_edge((unsigned)e.first, (unsigned)e.second);
    }
  }

  // step 3: compute orbit info by accumulating unions from automorphisms.
  DSU orbit_dsu((int)graph.num_vertices);
  bliss::Stats stats;

  // bliss calls report(n, aut) for each discovered automorphism, with aut being
  // a permutation array of length n, mapping i -> aut[i].
  //
  // we union i and aut[i] for binary gadget vertices only.
  auto report_automorphism = [&](unsigned int n, const unsigned int* aut) {
    if (aut == nullptr) return;
    if (n != graph.num_vertices) return;
    for (unsigned int i = 0; i < n; ++i) {
      unsigned int j = aut[i];
      if (j >= n) return;
      if (graph.is_binary_vertex[i] && graph.is_binary_vertex[j]) {
        orbit_dsu.unite((int)i, (int)j);
      }
    }
  };

  // step 4: optional termination callback based on wall clock time.
  const auto start = std::chrono::steady_clock::now();
  auto terminate_cb = [&]() -> bool {
    if (time_limit_sec <= 0.0) return false;
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (elapsed >= time_limit_sec) {
      terminated = true;
      return true; // tell bliss to stop
    }
    return false;
  };

  bliss_graph.find_automorphisms(stats, report_automorphism, terminate_cb);

  // path-compress all DSU roots (not strictly necessary, but keeps behavior consistent).
  for (uint32_t i = 0; i < graph.num_vertices; ++i) (void)orbit_dsu.find((int)i);

  // seed’s orbit root:
  int seed_root = orbit_dsu.find((int)seed_vertex);

  // orbit is reported as the set of binary gadget vertices in the same DSU component.
  for (uint32_t bin_pos = 0; bin_pos < graph.num_binary_vertices; ++bin_pos) {
    uint32_t vtx = graph.bin_vertex_ids[bin_pos];
    if (vtx < graph.num_vertices && orbit_dsu.find((int)vtx) == seed_root) {
      orbit_vertices_out.push_back(vtx);
    }
  }

  if (orbit_vertices_out.empty()) orbit_vertices_out.push_back(seed_vertex);

  std::sort(orbit_vertices_out.begin(), orbit_vertices_out.end());
  orbit_vertices_out.erase(std::unique(orbit_vertices_out.begin(), orbit_vertices_out.end()),
                           orbit_vertices_out.end());
  return true;
}

// --------------------------- reps query core ---------------------------
// compatibility endpoint: returns representative DSU root id for each binary gadget vertex.
// (parent code may skip using this now; still implemented.)
static bool build_and_solve_reps(const GraphData& graph,
                                 const std::vector<int8_t>& fixing_by_binpos,
                                 double time_limit_sec,
                                 bool& terminated,
                                 std::vector<uint32_t>& reps_out) {
  terminated = false;

  reps_out.clear();
  reps_out.resize(graph.num_binary_vertices, 0);

  if (graph.num_vertices == 0 || graph.num_binary_vertices == 0) return true;
  if (fixing_by_binpos.size() != graph.num_binary_vertices) return true;

  // same color refinement as orbit query.
  std::vector<uint32_t> refined_color = graph.base_color;
  for (uint32_t bin_pos = 0; bin_pos < graph.num_binary_vertices; ++bin_pos) {
    uint32_t vtx = graph.bin_vertex_ids[bin_pos];
    if (vtx >= graph.num_vertices) continue;

    int8_t fix_state = fixing_by_binpos[bin_pos];
    uint32_t bucket = 0;
    if (fix_state == 0) bucket = 1;
    else if (fix_state == 1) bucket = 2;

    refined_color[vtx] = refined_color[vtx] * 4u + bucket;
  }

  bliss::Graph bliss_graph;
  for (uint32_t i = 0; i < graph.num_vertices; ++i) {
    bliss_graph.add_vertex((unsigned)refined_color[i]);
  }
  for (const auto& e : graph.edges) {
    if (e.first < graph.num_vertices && e.second < graph.num_vertices && e.first != e.second) {
      bliss_graph.add_edge((unsigned)e.first, (unsigned)e.second);
    }
  }

  DSU orbit_dsu((int)graph.num_vertices);
  bliss::Stats stats;

  auto report_automorphism = [&](unsigned int n, const unsigned int* aut) {
    if (aut == nullptr) return;
    if (n != graph.num_vertices) return;
    for (unsigned int i = 0; i < n; ++i) {
      unsigned int j = aut[i];
      if (j >= n) return;
      if (graph.is_binary_vertex[i] && graph.is_binary_vertex[j]) {
        orbit_dsu.unite((int)i, (int)j);
      }
    }
  };

  const auto start = std::chrono::steady_clock::now();
  auto terminate_cb = [&]() -> bool {
    if (time_limit_sec <= 0.0) return false;
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (elapsed >= time_limit_sec) {
      terminated = true;
      return true;
    }
    return false;
  };

  bliss_graph.find_automorphisms(stats, report_automorphism, terminate_cb);

  for (uint32_t i = 0; i < graph.num_vertices; ++i) (void)orbit_dsu.find((int)i);

  // representative for each binary gadget vertex is its DSU root id.
  for (uint32_t bin_pos = 0; bin_pos < graph.num_binary_vertices; ++bin_pos) {
    uint32_t vtx = graph.bin_vertex_ids[bin_pos];
    if (vtx < graph.num_vertices) reps_out[bin_pos] = (uint32_t)orbit_dsu.find((int)vtx);
  }
  return true;
}

// --------------------------- main: protocol loop ---------------------------
int main(int argc, char** argv) {
  int bg_fd = -1;

  // parse CLI: --bg-fd FD
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--bg-fd" && i + 1 < argc) bg_fd = std::atoi(argv[++i]);
  }
  if (bg_fd < 0) return 2;

  GraphData graph;
  try {
    ::lseek(bg_fd, 0, SEEK_SET);
    graph = load_graph_from_fd(bg_fd);
  } catch (...) {
    return 3;
  }

  // persistent request loop: read req_type then dispatch.
  while (true) {
    uint8_t req_type = 0;
    if (!read_all(STDIN_FILENO, &req_type, sizeof(req_type))) return 0;

    if (req_type == 0) {
      // ---------------- orbit query ----------------
      uint32_t seed = 0;
      double time_limit_sec = 0.0;

      if (!read_all(STDIN_FILENO, &seed, sizeof(seed))) return 0;
      if (!read_all(STDIN_FILENO, &time_limit_sec, sizeof(time_limit_sec))) return 0;

      std::vector<int8_t> fixing_by_binpos(graph.num_binary_vertices, (int8_t)-1);
      if (graph.num_binary_vertices > 0) {
        if (!read_all(STDIN_FILENO, fixing_by_binpos.data(),
                      (size_t)graph.num_binary_vertices * sizeof(int8_t))) {
          return 0;
        }
      }

      uint8_t ok = 1, term = 0;
      std::vector<uint32_t> orbit_vertices;

      try {
        bool terminated = false;
        build_and_solve_orbit(graph, seed, fixing_by_binpos, time_limit_sec, terminated, orbit_vertices);
        term = terminated ? 1 : 0;
      } catch (...) {
        ok = 0;
        term = 0;
        orbit_vertices.clear();
      }

      uint32_t k = (uint32_t)orbit_vertices.size();
      if (!write_all(STDOUT_FILENO, &ok, sizeof(ok))) return 0;
      if (!write_all(STDOUT_FILENO, &term, sizeof(term))) return 0;

      if (ok != 1) {
        uint32_t z = 0;
        if (!write_all(STDOUT_FILENO, &z, sizeof(z))) return 0;
        continue;
      }

      if (!write_all(STDOUT_FILENO, &k, sizeof(k))) return 0;
      if (k > 0 && !write_all(STDOUT_FILENO, orbit_vertices.data(), (size_t)k * sizeof(uint32_t))) return 0;
    } else if (req_type == 1) {
      // ---------------- reps query (compatibility) ----------------
      double time_limit_sec = 0.0;
      if (!read_all(STDIN_FILENO, &time_limit_sec, sizeof(time_limit_sec))) return 0;

      std::vector<int8_t> fixing_by_binpos(graph.num_binary_vertices, (int8_t)-1);
      if (graph.num_binary_vertices > 0) {
        if (!read_all(STDIN_FILENO, fixing_by_binpos.data(),
                      (size_t)graph.num_binary_vertices * sizeof(int8_t))) {
          return 0;
        }
      }

      uint8_t ok = 1, term = 0;
      std::vector<uint32_t> reps;

      try {
        bool terminated = false;
        build_and_solve_reps(graph, fixing_by_binpos, time_limit_sec, terminated, reps);
        term = terminated ? 1 : 0;
      } catch (...) {
        ok = 0;
        term = 0;
        reps.clear();
      }

      uint32_t nb = (uint32_t)reps.size();
      if (!write_all(STDOUT_FILENO, &ok, sizeof(ok))) return 0;
      if (!write_all(STDOUT_FILENO, &term, sizeof(term))) return 0;

      if (ok != 1) {
        uint32_t z = 0;
        if (!write_all(STDOUT_FILENO, &z, sizeof(z))) return 0;
        continue;
      }

      if (!write_all(STDOUT_FILENO, &nb, sizeof(nb))) return 0;
      if (nb > 0 && !write_all(STDOUT_FILENO, reps.data(), (size_t)nb * sizeof(uint32_t))) return 0;
    } else {
      // unknown request type: respond with ok=0, term=0, size=0
      uint8_t ok = 0, term = 0;
      uint32_t z = 0;
      if (!write_all(STDOUT_FILENO, &ok, sizeof(ok))) return 0;
      if (!write_all(STDOUT_FILENO, &term, sizeof(term))) return 0;
      if (!write_all(STDOUT_FILENO, &z, sizeof(z))) return 0;
    }
  }
}
