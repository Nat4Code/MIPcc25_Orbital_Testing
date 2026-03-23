#!/usr/bin/env python3
"""
Benchmark script: heuristic_optimal vs Gurobi presolve

Outputs:
- primal.csv: per-case summary + winner by abs(PI)
- plots/plot_<case>.png: per-case incumbent trace (x starts at X_START)
- traces/trace_<case>.csv: time/objective points for gurobi + heuristic
- plots/primal_integral_summary_log.png: log-scaled |PI|
- plots/primal_integral_summary_share.png: normalized share in [0,1]

Notes:
- We clamp plot x-axis to start at X_START seconds so "late first incumbent"
  doesn't blow out plot scaling.
- We record incumbents via both MIPSOL and MIP callbacks (MIP_OBJBST), so
  the Gurobi trace is much less likely to be empty.
"""

import sys, csv, json, subprocess, math
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path.cwd()

# -------------------- knobs --------------------
TIME_LIMIT = 360            # horizon for primal integral computation
TIME_LIMIT_GUROBI = 10      # Gurobi solve cap (seconds)
TIME_LIMIT_HEUR = 10        # heuristic_optimal runtime cap (seconds)

X_START = 0.2               # plot x-axis starts at this time (seconds)
X_END = TIME_LIMIT_GUROBI   # per-case plot window
MIP_POLL_EVERY = 0.05       # seconds; how often to sample MIP_OBJBST
# ------------------------------------------------


def load_symmetry_cases(csv_path):
    cases = []
    with open(csv_path, 'r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            sym_ok = row.get('sym_ok', '').strip().lower() == 'true'
            if not sym_ok:
                continue
            case_id = int(row['case_id'])
            mps = row['mps']
            cases.append((case_id, mps))
    return cases


def primal_integral(times, objs, horizon=TIME_LIMIT):
    """Integrate objective over time using incumbent events (step function)."""
    if not times:
        return None
    combined = sorted(zip(times, objs))
    integral = 0.0
    for (t0, o0), (t1, o1) in zip(combined, combined[1:]):
        integral += (t1 - t0) * o0
    integral += (horizon - combined[-1][0]) * combined[-1][1]
    return integral


def sanitize_trace(times, objs, x_start=X_START, x_end=X_END):
    """Keep events within [x_start, x_end] for plotting; clamp early events."""
    if not times or not objs:
        return [], []

    cleaned = [(t, o) for t, o in zip(times, objs) if t <= x_end]
    if not cleaned:
        return [], []

    cleaned.sort(key=lambda p: p[0])

    out = []
    for (t, o) in cleaned:
        if t < x_start:
            # collapse everything before x_start into one point at x_start
            if out and abs(out[-1][0] - x_start) < 1e-12:
                out[-1] = (x_start, o)
            else:
                out.append((x_start, o))
        else:
            out.append((t, o))

    # de-dupe equal timestamps (keep last)
    dedup = []
    for t, o in out:
        if dedup and abs(dedup[-1][0] - t) < 1e-12:
            dedup[-1] = (t, o)
        else:
            dedup.append((t, o))

    return [t for t, _ in dedup], [o for _, o in dedup]


def run_gurobi_with_presolve(mps_path):
    """
    Run Gurobi MIP with presolve enabled, record an incumbent trace.

    Key change:
    - We record from MIPSOL (exact incumbent events)
    - AND from MIP using MIP_OBJBST polling (best incumbent so far)
      so the blue line doesn't disappear when MIPSOL events are sparse.
    """
    model = gp.read(mps_path)
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', TIME_LIMIT_GUROBI)
    model.setParam('Presolve', 2)

    times = []
    objs = []

    last_obj = None
    last_poll_t = 0.0

    def record(t, obj):
        nonlocal last_obj
        if obj is None:
            return
        try:
            obj = float(obj)
            t = float(t)
        except Exception:
            return

        # ignore infinities
        if not math.isfinite(obj) or not math.isfinite(t):
            return

        # only record improvements / changes
        if last_obj is None or abs(obj - last_obj) > 1e-12:
            times.append(t)
            objs.append(obj)
            last_obj = obj

    def cb(m, where):
        nonlocal last_poll_t
        if where == GRB.Callback.MIPSOL:
            t = m.cbGet(GRB.Callback.RUNTIME)
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            record(t, obj)

        elif where == GRB.Callback.MIP:
            # Poll OBJBST periodically, so we get a trace even without MIPSOL spam
            t = m.cbGet(GRB.Callback.RUNTIME)
            if t - last_poll_t >= MIP_POLL_EVERY:
                last_poll_t = t
                objbst = m.cbGet(GRB.Callback.MIP_OBJBST)
                # When no incumbent exists, OBJBST can be +/-inf
                record(t, objbst)

    model.optimize(cb)

    final_obj = None
    if model.SolCount > 0:
        final_obj = float(model.ObjVal)

    # If we found a solution but still have no trace points (rare), force one point.
    if final_obj is not None and not times:
        times = [float(model.Runtime)]
        objs = [final_obj]

    return times, objs, final_obj


def run_heuristic_optimal(mps_path):
    """Run heuristic_optimal and extract solution from JSON output."""
    cmd = [str(ROOT / 'heuristic_optimal'), mps_path]
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=TIME_LIMIT_HEUR
        )
        out = p.stdout + '\n' + p.stderr

        for line in out.splitlines():
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and 'heuristic_obj' in obj:
                    return obj.get('heuristic_time_sec'), obj.get('heuristic_obj')
            except json.JSONDecodeError:
                pass
        return None, None
    except Exception as e:
        print(f'  heuristic_optimal failed: {e}', file=sys.stderr)
        return None, None


def write_trace_csv(case_id, pre_times, pre_objs, opt_time, opt_obj):
    """Write per-case time/objective points to traces/trace_<case_id>.csv."""
    traces_dir = ROOT / 'traces'
    traces_dir.mkdir(exist_ok=True)
    path = traces_dir / f"trace_{case_id}.csv"

    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'time_sec', 'objective'])
        for t, o in sorted(zip(pre_times, pre_objs)):
            w.writerow(['gurobi_presolve', t, o])
        if opt_time is not None and opt_obj is not None:
            w.writerow(['heuristic_optimal', opt_time, opt_obj])


def plot_case_trace(case_id, pre_times, pre_objs, opt_time, opt_obj):
    """
    Per-case plot:
    - Step curve for Gurobi presolve incumbents (within [X_START, X_END])
    - One point for heuristic_optimal
    """
    plt.figure(figsize=(10, 6))

    st, so = sanitize_trace(pre_times, pre_objs, X_START, X_END)
    if st and so:
        # Ensure step starts at X_START
        plt.step([X_START] + st, [so[0]] + so, where='post',
                 label='Gurobi presolve', linewidth=2)
    else:
        # Make it explicit when there is no incumbent (so you know why it's empty)
        plt.text(0.5, 0.5, "No Gurobi incumbent in time limit",
                 transform=plt.gca().transAxes,
                 ha='center', va='center', fontsize=10, alpha=0.8)

    if opt_obj is not None and opt_time is not None:
        tplot = float(opt_time)
        if tplot < X_START:
            tplot = X_START
        if tplot > X_END:
            # still show it at right edge so it doesn't vanish
            tplot = X_END
        plt.plot([tplot], [opt_obj], 's', label='heuristic_optimal', markersize=8)

    plt.xlim(X_START, X_END)
    plt.xlabel('time (s)', fontsize=11)
    plt.ylabel('objective', fontsize=11)
    plt.title(f'Test {case_id}: heuristic_optimal vs presolve', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plots_dir = ROOT / 'plots'
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f'plot_{case_id}.png', dpi=120, bbox_inches='tight')
    plt.close()


def plot_pi_summary_log(rows):
    """Summary bar chart using log10(|PI| + 1) to tame the scale."""
    cases = []
    opt_log = []
    pre_log = []

    for r in rows:
        cid = r['case_id']
        pi_opt = r['pi_opt']
        pi_pre = r['pi_pre']
        if pi_opt is None and pi_pre is None:
            continue

        a = abs(pi_opt) if pi_opt is not None else 0.0
        b = abs(pi_pre) if pi_pre is not None else 0.0
        cases.append(cid)
        opt_log.append(math.log10(a + 1.0))
        pre_log.append(math.log10(b + 1.0))

    if not cases:
        return

    import numpy as np
    x = np.arange(len(cases))
    width = 0.42

    plt.figure(figsize=(max(10, 0.35 * len(cases)), 6))
    plt.bar(x - width/2, opt_log, width, label='log10(|PI|+1) heuristic_optimal')
    plt.bar(x + width/2, pre_log, width, label='log10(|PI|+1) Gurobi presolve')

    plt.xticks(x, [str(c) for c in cases], rotation=90)
    plt.ylabel('log10(|primal integral| + 1)', fontsize=11)
    plt.title('Primal integral magnitude (log-scaled)', fontsize=12)
    plt.grid(True, axis='y', alpha=0.25)
    plt.legend(fontsize=10)

    plots_dir = ROOT / 'plots'
    plots_dir.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(plots_dir / 'primal_integral_summary_log.png', dpi=140, bbox_inches='tight')
    plt.close()


def plot_pi_summary_share(rows):
    """
    Summary plot in [0,1]:
    share = |PI|_heur / (|PI|_heur + |PI|_gurobi)
    """
    cases = []
    shares = []

    for r in rows:
        cid = r['case_id']
        pi_opt = r['pi_opt']
        pi_pre = r['pi_pre']
        if pi_opt is None and pi_pre is None:
            continue

        a = abs(pi_opt) if pi_opt is not None else 0.0
        b = abs(pi_pre) if pi_pre is not None else 0.0
        denom = a + b
        share = (a / denom) if denom > 0 else 0.5

        cases.append(cid)
        shares.append(share)

    if not cases:
        return

    plt.figure(figsize=(max(10, 0.35 * len(cases)), 4.8))
    plt.plot(range(len(cases)), shares, marker='o', linestyle='none', markersize=4)
    plt.axhline(0.5, linewidth=1, alpha=0.5)
    plt.ylim(-0.02, 1.02)

    plt.xticks(range(len(cases)), [str(c) for c in cases], rotation=90)
    plt.ylabel('share of |PI| from heuristic_optimal', fontsize=11)
    plt.title('Normalized |PI| share (0 = all Gurobi, 1 = all heuristic)', fontsize=12)
    plt.grid(True, axis='y', alpha=0.25)

    plots_dir = ROOT / 'plots'
    plots_dir.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(plots_dir / 'primal_integral_summary_share.png', dpi=140, bbox_inches='tight')
    plt.close()


def main():
    cases_csv = ROOT / 'cases_with_symmetry_real.csv'
    cases = load_symmetry_cases(cases_csv)
    if not cases:
        print('ERROR: no symmetry cases found in cases_with_symmetry_real.csv', file=sys.stderr)
        sys.exit(1)

    print(f'Found {len(cases)} symmetry cases')

    primal_csv = ROOT / 'primal.csv'
    summary_rows = []

    with open(primal_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'case_id',
            'heuristic_optimal_time', 'heuristic_optimal_obj',
            'heuristic_optimal_integral', 'heuristic_optimal_abs_integral',
            'gurobi_presolve_last_inc_time', 'gurobi_presolve_final_obj',
            'gurobi_presolve_integral', 'gurobi_presolve_abs_integral',
            'better_abs_integral'
        ])

        for case_id, mps in cases:
            print(f'Processing case {case_id}...')
            mps_full = str(ROOT / mps)

            # Run Gurobi with presolve (trace)
            pre_times, pre_objs, pre_final = run_gurobi_with_presolve(mps_full)
            pre_int = primal_integral(pre_times, pre_objs, horizon=TIME_LIMIT)
            pre_abs = abs(pre_int) if pre_int is not None else None

            # Run heuristic_optimal
            opt_time, opt_obj = run_heuristic_optimal(mps_full)

            opt_int = None
            if opt_obj is not None and opt_time is not None:
                # Single incumbent at opt_time that persists to horizon
                t = float(opt_time)
                o = float(opt_obj)
                opt_int = t * o + (TIME_LIMIT - t) * o
            opt_abs = abs(opt_int) if opt_int is not None else None

            # Winner by abs integral
            winner = 'tie/na'
            if opt_abs is not None or pre_abs is not None:
                if opt_abs is None:
                    winner = 'gurobi_presolve'
                elif pre_abs is None:
                    winner = 'heuristic_optimal'
                else:
                    if opt_abs > pre_abs:
                        winner = 'heuristic_optimal'
                    elif pre_abs > opt_abs:
                        winner = 'gurobi_presolve'
                    else:
                        winner = 'tie/na'

            # Write trace CSV (time/objective points)
            write_trace_csv(case_id, pre_times, pre_objs, opt_time, opt_obj)

            # Plot per-case trace
            plot_case_trace(case_id, pre_times, pre_objs, opt_time, opt_obj)

            # Write summary CSV row
            last_inc_t = pre_times[-1] if pre_times else None
            w.writerow([
                case_id,
                opt_time, opt_obj,
                opt_int, opt_abs,
                last_inc_t, pre_final,
                pre_int, pre_abs,
                winner
            ])

            summary_rows.append({'case_id': case_id, 'pi_opt': opt_int, 'pi_pre': pre_int})

            print(f'  heuristic_optimal: t={opt_time}, obj={opt_obj}, PI={opt_int} (|PI|={opt_abs})')
            print(f'  gurobi_presolve: final_obj={pre_final}, PI={pre_int} (|PI|={pre_abs})')
            print(f'  better (by |PI|): {winner}')

    # Summary plots with sane scaling
    plot_pi_summary_log(summary_rows)
    plot_pi_summary_share(summary_rows)

    print(f'\nDone. Results in {primal_csv}')
    print(f'- Per-case traces: plots/plot_<case>.png (x starts at {X_START}s)')
    print('- Per-case trace CSVs: traces/trace_<case>.csv')
    print('- PI summary (log): plots/primal_integral_summary_log.png')
    print('- PI summary (share): plots/primal_integral_summary_share.png')


if __name__ == '__main__':
    main()