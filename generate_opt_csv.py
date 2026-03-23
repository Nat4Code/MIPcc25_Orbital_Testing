#!/usr/bin/env python3
"""
generate_opt_csv.py

Run heuristic_optimal on all cases with sym_ok=True from cases_with_symmetry_real.csv
and write a simple CSV with two columns: time, obj.

- Time and obj come from the JSON line emitted by heuristic_optimal:
  {"heuristic_time_sec": ..., "heuristic_obj": ...}

Usage:
  python3 generate_opt_csv.py
  python3 generate_opt_csv.py --cases cases_with_symmetry_real.csv --out opt_time_obj.csv
  python3 generate_opt_csv.py --time-limit 10
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()


def load_symmetry_mps_paths(cases_csv: Path):
    """Return list of absolute MPS paths for rows where sym_ok=True."""
    out = []
    with open(cases_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sym_ok = (row.get("sym_ok", "").strip().lower() == "true")
            if not sym_ok:
                continue
            mps = row.get("mps", "").strip()
            if not mps:
                continue
            out.append((row.get("case_id", "").strip(), str((ROOT / mps).resolve())))
    return out


def run_heuristic_optimal(exe: Path, mps_path: str, timeout_s: float):
    """
    Run heuristic_optimal and parse JSON output for heuristic_time_sec and heuristic_obj.
    Returns (time, obj) or (None, None).
    """
    cmd = [str(exe), mps_path]
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return None, None
    except Exception:
        return None, None

    # heuristic_optimal sometimes prints timings to stderr; JSON to stdout.
    # We'll scan both, line-by-line, and take the first dict with heuristic_obj.
    blob = (p.stdout or "") + "\n" + (p.stderr or "")
    for line in blob.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and ("heuristic_obj" in obj or "heuristic_time_sec" in obj):
            t = obj.get("heuristic_time_sec", None)
            o = obj.get("heuristic_obj", None)
            # Coerce to float if possible
            try:
                t = float(t) if t is not None else None
            except Exception:
                t = None
            try:
                o = float(o) if o is not None else None
            except Exception:
                o = None
            return t, o

    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="cases_with_symmetry_real.csv",
                    help="CSV containing case_id,mps,sym_ok columns (default: cases_with_symmetry_real.csv)")
    ap.add_argument("--out", default="opt_time_obj.csv",
                    help="Output CSV (default: opt_time_obj.csv)")
    ap.add_argument("--exe", default=str(ROOT / "heuristic_optimal"),
                    help="Path to heuristic_optimal executable (default: ./heuristic_optimal)")
    ap.add_argument("--time-limit", type=float, default=10.0,
                    help="Per-run timeout seconds (default: 10.0)")
    args = ap.parse_args()

    cases_csv = Path(args.cases)
    if not cases_csv.exists():
        print(f"ERROR: cases CSV not found: {cases_csv}", file=sys.stderr)
        return 1

    exe = Path(args.exe)
    if not exe.exists():
        print(f"ERROR: heuristic_optimal executable not found: {exe}", file=sys.stderr)
        return 2

    cases = load_symmetry_mps_paths(cases_csv)
    if not cases:
        print("ERROR: no sym_ok=True cases found.", file=sys.stderr)
        return 3

    out_path = Path(args.out)

    # Write ONLY two columns: time, obj (as requested).
    # (No headers beyond these two.)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "obj"])
        for case_id, mps_path in cases:
            t, o = run_heuristic_optimal(exe, mps_path, args.time_limit)
            # If it fails, leave blanks (or you can change to "NA")
            w.writerow([t if t is not None else "", o if o is not None else ""])
            print(f"case {case_id}: time={t}, obj={o}")

    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())