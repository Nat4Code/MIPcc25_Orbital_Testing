import sys, csv, json, subprocess, os
from pathlib import Path

# ----------------- process runner -----------------

def _to_text(x):
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)

def run_cmd(cmd, timeout_sec, env=None, cwd=None):
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            env=env,
            cwd=cwd,
        )
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        return p.returncode, out, False
    except subprocess.TimeoutExpired as e:
        out = _to_text(getattr(e, "stdout", None)) + "\n" + _to_text(getattr(e, "stderr", None))
        return 124, out, True


# ----------------- JSON extraction (LINE-BASED; handles nested braces) -----------------

def parse_json_lines(text):
    objs = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if not (s.startswith("{") and s.endswith("}")):
            continue
        try:
            o = json.loads(s)
            if isinstance(o, dict):
                objs.append(o)
        except Exception:
            continue
    return objs

def extract_last_tool_json(text, expected_tool=None):
    """
    Prefer the last JSON line for the expected tool.
    Fall back to last JSON line overall.
    """
    objs = parse_json_lines(text)
    if not objs:
        return None

    if expected_tool is not None:
        filt = [o for o in objs if o.get("tool") == expected_tool]
        if filt:
            objs = filt

    # Prefer objects that look "final"
    def is_finalish(o):
        return (
            ("ok" in o and "phase" not in o) or
            ("best_obj" in o) or
            ("tight_obj" in o) or
            ("signal" in o) or
            ("error_type" in o)
        )

    finals = [o for o in objs if is_finalish(o)]
    return finals[-1] if finals else objs[-1]

# ----------------- value helpers -----------------

def get_float(d, k):
    try:
        v = d.get(k, None)
        return None if v is None else float(v)
    except Exception:
        return None

def get_bool(d, k):
    v = d.get(k, None)
    if v is None:
        return None
    return bool(v)

def pick_obj_orbital(d):
    # orbital + baseline use best_obj
    v = d.get("best_obj", None)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None

def pick_obj_py(d):
    # qp_bench.py prints tight_obj
    v = d.get("tight_obj", None)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None

def fmt_time(x):
    return "None" if x is None else f"{x:.6f}"

def fmt_obj(x):
    return "None" if x is None else f"{x:.6f}"

# ----------------- symmetry case loader -----------------

def load_symmetry_cases(csv_path):
    cases = []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sym_ok = row.get("sym_ok", "").strip().lower() == "true"
            if not sym_ok:
                continue
            case_id = int(row["case_id"])
            mps = row["mps"]
            orbit_cnt = row.get("sym_orbit_count", "")
            sym_orbit_count = int(orbit_cnt) if orbit_cnt.strip() != "" else None
            cases.append((case_id, mps, sym_orbit_count))
    return cases

# ----------------- main -----------------

def main():
    root = Path.cwd()

    cases_csv = root / "cases_with_symmetry.csv"
    if not cases_csv.exists():
        print(f"Missing {cases_csv}", file=sys.stderr)
        sys.exit(1)

    cases = load_symmetry_cases(cases_csv)
    if not cases:
        print("No symmetry cases with sym_ok=True found.", file=sys.stderr)
        sys.exit(1)

    orbital_bin  = str((root / "orbital_branch_local").resolve())
    baseline_bin = str((root / "baseline").resolve())
    py_script    = str((root / "qp_bench.py").resolve())

    # Params
    time_limit = 300
    extra_slack = 30

    # Make sure local bliss is used
    env = os.environ.copy()
    local_bliss_lib = str((root / "local" / "bliss" / "lib").resolve())
    env["LD_LIBRARY_PATH"] = local_bliss_lib + ":" + env.get("LD_LIBRARY_PATH", "")

    out_csv = root / "results_symmetry_only.csv"
    fieldnames = [
        "case_id", "mps", "sym_orbit_count",

        "orbital_ok", "orbital_timeout", "orbital_rc", "orbital_time", "orbital_obj",
        "baseline_ok", "baseline_timeout", "baseline_rc", "baseline_time", "baseline_obj",
        "py_ok", "py_timeout", "py_rc", "py_time", "py_obj",
    ]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for idx, (case_id, mps_path, sym_orbit_count) in enumerate(cases, start=1):
            mps_path = str(Path(mps_path))
            short = Path(mps_path).name

            row = {
                "case_id": case_id,
                "mps": mps_path,
                "sym_orbit_count": sym_orbit_count,
            }

            # ---- ORBITAL ----
            orbital_cmd = [orbital_bin, mps_path, "--time-limit", str(time_limit), "--threads", "1"]
            rc, out, to = run_cmd(orbital_cmd, timeout_sec=time_limit + extra_slack, env=env, cwd=str(root))
            j = extract_last_tool_json(out, expected_tool="orbital_branch_exact_bliss") or {}
            row["orbital_rc"] = rc
            row["orbital_timeout"] = to
            row["orbital_ok"] = get_bool(j, "ok")
            row["orbital_time"] = get_float(j, "wall_runtime_sec")
            row["orbital_obj"] = pick_obj_orbital(j)
            if to:
                row["orbital_obj"] = None

            # ---- BASELINE ----
            baseline_cmd = [baseline_bin, mps_path, str(time_limit)]
            rc, out, to = run_cmd(baseline_cmd, timeout_sec=time_limit + extra_slack, env=env, cwd=str(root))
            j = extract_last_tool_json(out, expected_tool="baseline") or {}
            row["baseline_rc"] = rc
            row["baseline_timeout"] = to
            row["baseline_ok"] = get_bool(j, "ok")
            row["baseline_time"] = get_float(j, "wall_runtime_sec")
            row["baseline_obj"] = pick_obj_orbital(j)
            if to:
                row["baseline_obj"] = None

            # ---- PY ----
            py_cmd = [sys.executable, py_script, mps_path]
            rc, out, to = run_cmd(py_cmd, timeout_sec=time_limit + extra_slack, env=env, cwd=str(root))
            j = extract_last_tool_json(out, expected_tool="qp_py") or extract_last_tool_json(out, expected_tool=None) or {}
            row["py_rc"] = rc
            row["py_timeout"] = to
            row["py_ok"] = get_bool(j, "ok")
            row["py_time"] = get_float(j, "wall_runtime_sec")
            row["py_obj"] = pick_obj_py(j)
            if to:
                row["py_obj"] = None

            w.writerow(row)

            print(
                f"[{idx:03d}/{len(cases):03d}] {short} | "
                f"orbital: t={fmt_time(row['orbital_time'])} obj={fmt_obj(row['orbital_obj'])} | "
                f"baseline: t={fmt_time(row['baseline_time'])} obj={fmt_obj(row['baseline_obj'])} | "
                f"py: t={fmt_time(row['py_time'])} obj={fmt_obj(row['py_obj'])}"
            )

    print(f"Wrote {out_csv}")

if __name__ == "__main__":
    main()
