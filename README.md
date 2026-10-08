# Symmetry-Aware Branching for Mixed-Integer Optimization

Research code for testing how model symmetry can guide branching and primal heuristics in mixed-integer optimization. The core implementation combines **C++20, Gurobi, and Bliss**, with graph encodings for linear and quadratic constraints, parallel node search, and Python benchmarking tools.

The central question is practical: **when does exploiting symmetry save more search effort than it costs to detect and maintain it?** This repository compares graph-based branching, depth-limited Gurobi handoffs, and partition-guided heuristics against direct Gurobi solves.

**Research status:** manuscript in preparation. The public repository is an experimental implementation; see [current scope and validation notes](#current-scope-and-validation-notes) before interpreting solver names or benchmark results as correctness or performance guarantees.

## Highlights

- **Model structure represented as colored graphs.** Variables, linear and quadratic constraints, coefficient occurrences, and quadratic product terms form the graph used for symmetry detection.
- **Bliss automorphisms under node fixings.** A persistent helper process computes binary-variable orbits after distinguishing free, zero-fixed, and one-fixed variables.
- **Parallel search with separate solver state.** Each worker owns a Gurobi environment, model state, and symmetry helper. The graph is passed through a Linux memory-backed file descriptor.
- **Configurable handoff to Gurobi.** Depth-limited variants explore the upper part of the tree before handing remaining subproblems to Gurobi.
- **A reproducible experimental starting point.** The repository includes 95 MPS models, a 30-case symmetry subset, JSON solver output, CSV summaries, and incumbent-trace plotting tools.

## Repository map

| File or directory | Purpose |
| --- | --- |
| [`orbit_branch_exact_bliss.cpp`](orbit_branch_exact_bliss.cpp) | Parallel custom branching driver with Bliss orbit queries and Gurobi subproblem handoffs |
| [`orbit_branch_depth_bliss.cpp`](orbit_branch_depth_bliss.cpp) | Depth-limited variant; default handoff depth is 2 |
| [`orbit_branch_sensing_bliss.cpp`](orbit_branch_sensing_bliss.cpp) | Currently the same implementation as the depth variant, with default handoff depth 3 |
| [`orbit_helper.cpp`](orbit_helper.cpp) | Persistent Bliss helper; communicates with the driver through a binary protocol |
| [`baseline_solve.c`](baseline_solve.c) | Direct Gurobi C API baseline with a final JSON result |
| [`qp_bench.py`](qp_bench.py) | Python/NetworkX prototype for graph refinement, folding, and variable-fixing experiments |
| [`heuristic_optimal.cpp`](heuristic_optimal.cpp) | Partition-guided candidate construction followed by a short Gurobi solve using a MIP start |
| `heuristic*.cpp` | Related heuristic experiments and variants |
| [`heuristic_with_cut.cpp`](heuristic_with_cut.cpp), [`orbit_exact_with_cut.cpp`](orbit_exact_with_cut.cpp) | Additional solve/cut experiments; symmetry constraints require separate correctness validation |
| [`Initial_problem_set/`](Initial_problem_set/) | 95 MPS instances, `Test_prob_1.mps` through `Test_prob_95.mps` |
| [`Problem_set_solutions/`](Problem_set_solutions/) | Stored reference-solution files |
| [`cases_with_symmetry_real.csv`](cases_with_symmetry_real.csv) | 30-case symmetry subset and recorded orbit counts |
| [`benchmark.py`](benchmark.py) | Depth, sensing, Python-prototype, and baseline comparisons |
| [`benchmark_heuristic.py`](benchmark_heuristic.py) | Heuristic/presolve comparisons, incumbent traces, and plots |
| [`run_bench.py`](run_bench.py) | Earlier comparison harness with legacy file/binary names |
| `benchmark_results.csv`, `primal.csv`, `traces/`, `plots/` | Stored experimental outputs |

## Requirements

Run the C++ branching drivers on **Linux**: they use `memfd_create`, POSIX pipes, and process spawning.

- A C++20 compiler and a C compiler
- A licensed Gurobi installation with C and C++ headers/libraries
- Bliss headers and library compatible with the callback/termination interface used in `orbit_helper.cpp`
- The header-only `nlohmann/json` library for the branching drivers
- Python 3; the Python workflows additionally use `gurobipy`, `networkx`, `numpy`, and `matplotlib`

Gurobi and Bliss are external dependencies. The repository does not include their installations. Rebuild the tracked executables locally so the binaries match your libraries and the source being evaluated.

## Build

Clone the repository and work from its root:

```bash
git clone https://github.com/Nat4Code/MIPcc25_Orbital_Testing.git
cd MIPcc25_Orbital_Testing
```

Set the paths to your installations. These examples use Gurobi 12.0.3; change both the directory and library name for your version. For Gurobi 13, the link library is typically `gurobi130`.

```bash
export GUROBI_HOME=/opt/gurobi1203/linux64
export GUROBI_LINK_LIB=gurobi120
export BLISS_PREFIX="$PWD/local/bliss"
```

`BLISS_PREFIX` must point to an installed Bliss prefix containing `include/bliss/graph.hh`, `include/bliss/stats.hh`, and `lib/libbliss.*`. Adjust the `lib` directory below if your installation uses `lib64`. Make sure `nlohmann/json.hpp` is on the compiler's include path.

Build the helper:

```bash
g++ -O3 -DNDEBUG -std=c++20 \
  -I"$BLISS_PREFIX/include" \
  orbit_helper.cpp -o orbit_helper \
  -L"$BLISS_PREFIX/lib" -lbliss \
  -Wl,-rpath,"$BLISS_PREFIX/lib"
```

Build the three branching drivers:

```bash
for variant in exact depth sensing; do
  g++ -O3 -DNDEBUG -std=c++20 -pthread \
    -I"$GUROBI_HOME/include" \
    "orbit_branch_${variant}_bliss.cpp" -o "orbit_branch_${variant}" \
    -L"$GUROBI_HOME/lib" -lgurobi_c++ -l"$GUROBI_LINK_LIB" \
    -Wl,-rpath,"$GUROBI_HOME/lib"
done
```

Build the baseline and the heuristic used by `benchmark_heuristic.py`:

```bash
gcc -O3 -std=gnu11 \
  -I"$GUROBI_HOME/include" \
  baseline_solve.c -o baseline \
  -L"$GUROBI_HOME/lib" -l"$GUROBI_LINK_LIB" -lm \
  -Wl,-rpath,"$GUROBI_HOME/lib"

g++ -O3 -DNDEBUG -std=c++20 \
  -I"$GUROBI_HOME/include" \
  heuristic_optimal.cpp -o heuristic_optimal \
  -L"$GUROBI_HOME/lib" -lgurobi_c++ -l"$GUROBI_LINK_LIB" \
  -Wl,-rpath,"$GUROBI_HOME/lib"
```

If the helper fails to compile at `find_automorphisms`, check that your Bliss headers expose the overload used by `orbit_helper.cpp`; older interfaces differ. If Gurobi cannot be loaded at runtime, check that its headers, C++ library, versioned library, and runtime path all come from the same installation.

## Run one instance

Start with explicit, modest resource limits:

```bash
./orbit_branch_exact Initial_problem_set/Test_prob_4.mps \
  --time-limit 60 --node-limit 500 \
  --threads 1 --gurobi-threads 1 --sym-budget 0.35

./orbit_branch_depth Initial_problem_set/Test_prob_4.mps \
  --time-limit 60 --node-limit 500 \
  --threads 1 --gurobi-threads 1 --sym-budget 0.25 \
  --orbit-depth 2

./orbit_branch_sensing Initial_problem_set/Test_prob_4.mps \
  --time-limit 60 --node-limit 500 \
  --threads 1 --gurobi-threads 1 --sym-budget 0.25 \
  --orbit-depth 3

./baseline Initial_problem_set/Test_prob_4.mps 60
```

The branching drivers launch `./orbit_helper`, so run them from the directory containing that executable. The helper is a subprocess service, not a standalone `model.mps` command.

| Option | Meaning |
| --- | --- |
| `--time-limit T` | Driver wall-time budget in seconds; default `0` means no configured global limit |
| `--node-limit N` | Custom search-node limit; default `20000` |
| `--threads K` | Number of custom search workers; default `32` |
| `--gurobi-threads G` | Gurobi threads per worker; set explicitly to control total solver resources |
| `--sym-budget S` | Budget per orbit query; defaults are `0.35` for exact and `0.25` for depth/sensing |
| `--orbit-depth D` | Handoff depth for depth/sensing only; defaults are `2` and `3`, respectively |

The exact driver clamps Gurobi threads to at least one. Depth/sensing default to leaving Gurobi's thread setting unchanged. Use explicit settings for comparisons; the baseline currently has no thread-count CLI option. The wall-time limit is checked between operations and is not a strict process deadline for every relaxation or helper operation.

## Read the output

The branching drivers emit a startup JSON record and a final result record. Preserve the full output, then inspect the **final** record:

```bash
mkdir -p results
./orbit_branch_exact Initial_problem_set/Test_prob_4.mps \
  --time-limit 60 --node-limit 500 \
  --threads 1 --gurobi-threads 1 --sym-budget 0.35 \
  > results/test4.jsonl 2> results/test4.stderr.log

tail -n 1 results/test4.jsonl
```

Useful fields include `best_obj`, `have_incumbent`, `best_binary_solution`, `explored_nodes`, `branched_nodes`, `handoff_nodes`, `terminated_early`, and `wall_runtime_sec`. The baseline also reports its Gurobi status.

`ok: true` means the driver's result path completed; it does **not** certify optimality. Check termination, incumbent feasibility, model scope, and the validation notes below. A symmetry-query timeout can currently terminate the whole custom search.

## Python workflows

Create an environment and install the Python dependencies. Use a `gurobipy` release compatible with the Gurobi version you intend to compare.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install gurobipy networkx numpy matplotlib
```

The supplied scripts have different entry points and overwrite their named result files. Preserve existing outputs before launching a new campaign.

| Script | Setup and behavior |
| --- | --- |
| `python qp_bench.py Initial_problem_set/Test_prob_4.mps` | Runs the Python prototype on one model; some logic depends on variable-name conventions |
| `python benchmark.py` | Expects `cases_with_symmetry.csv`; compares depth, sensing, Python, and baseline with a 360-second cap per process |
| `python benchmark_heuristic.py` | Uses the supplied `cases_with_symmetry_real.csv` and `heuristic_optimal`; records CSVs, traces, and plots |
| `python run_bench.py` | Expects `cases_with_symmetry.csv` and a binary named `orbital_branch_local`; these are legacy names |

For the legacy case-file name, create a copy only if it is absent:

```bash
if [ ! -e cases_with_symmetry.csv ]; then
  cp cases_with_symmetry_real.csv cases_with_symmetry.csv
fi
```

For `run_bench.py`, change its `orbital_bin` path from `orbital_branch_local` to `orbit_branch_exact`, or build the selected driver under that name. A full campaign can take hours. Use a single-instance run first, and record the source revision, machine, library versions, thread counts, limits, and solver settings with any reported result.

## Stored results

The checked-in files document individual experimental runs, not a uniform speedup across the test set. For example, `benchmark_results.csv` records the following wall times at the same reported final objective within each row:

| Instance | Depth variant | Direct Gurobi | Reported objective |
| --- | ---: | ---: | ---: |
| `Test_prob_95` | 88.780 s | 247.923 s | 602 |
| `Test_prob_67` | 5.417 s | 6.331 s | 334 |
| `Test_prob_65` | 47.017 s | 1.289 s | 0 |

These are archived observations, not newly reproduced results or verified optimality claims. They show why the comparison matters: custom branching helps on some recorded cases and costs substantially more on others. The CSV alone does not establish matched hardware, equal thread budgets, repeated-run variability, feasibility, or a proof of optimality.

## Current scope and validation notes

The following details apply to the public source reviewed at commit [`78d238e`](https://github.com/Nat4Code/MIPcc25_Orbital_Testing/tree/78d238e883e098e2a678afcbd6378af0855cc2ec). They are material to interpreting the experiments:

1. **Branch coverage needs correction or a supporting model-specific proof.** The multi-variable orbit branch in the exact/depth/sensing drivers creates children with a zero prefix and one variable fixed to one. It does not create the all-zero orbit child. Without an additional condition excluding that assignment, the branch set can omit feasible solutions. The filename `exact` is not a correctness guarantee.
2. **The graph representation has a defined scope.** It encodes variable attributes, linear constraints, and quadratic constraints, with numeric keys rounded at a scale of `1e9`. It does not encode quadratic objective terms, SOS constraints, or general constraints. Bliss automorphisms of this graph are not automatically symmetries of model features that the graph omits. Incumbent comparisons in the custom drivers also assume minimization.
3. **A heuristic objective may not be a verified feasible incumbent.** If the short final solve in `heuristic_optimal.cpp` finds no solution, the program can report an objective computed from its candidate vector without a feasibility certificate. Validate that candidate before treating it as a solution.
4. **The current integral labels are exploratory.** `benchmark_heuristic.py` integrates raw objective values, uses a 360-second integration horizon with 10-second solve caps, backfills the heuristic's final objective to time zero, and labels the larger absolute integral as better. This is not a normalized primal-gap integral and does not support the stored winner labels as a general quality metric.
5. **Symmetry-cut experiments require separate validation.** `heuristic_with_cut.cpp` explicitly uses heuristic partitions rather than proven orbits. Even membership in a true orbit does not imply that every pairwise swap is a valid automorphism. Leave `--symbreak` and `--symcut` off unless the particular constraints/cuts have been justified.

Before making publication-level correctness or performance claims, address branch coverage, validate solutions against the original models, use model-sense-aware metrics, and rerun comparisons with matched resource budgets. These notes describe the public implementation; they do not assert that a newer manuscript or private revision has the same limitations.

## Author

**Nathaniel Sheets** — Data Science & Engineering, University of Tennessee, Knoxville

[GitHub](https://github.com/Nat4Code) · [LinkedIn](https://www.linkedin.com/in/nathaniel-sheets-362a8618a/)
