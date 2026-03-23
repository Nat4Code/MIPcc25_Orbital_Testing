#!/usr/bin/env python3
import sys, csv, json, subprocess, os, time
from pathlib import Path

ROOT = Path.cwd()

TIME_LIMIT = 360 # 6 min right now

def _to_text(x):
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)

def run_cmd(cmd, timeout_sec, env=None, cwd=None):
    try:
        t0 = time.time()
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            env=env,
            cwd=cwd,
        )
        elapsed = time.time() - t0
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        return p.returncode, out, False, elapsed
    except subprocess.TimeoutExpired as e:
        out = _to_text(getattr(e, "stdout", None)) + "\n" + _to_text(getattr(e, "stderr", None))
        return 124, out, True, float(timeout_sec)


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
    objs = parse_json_lines(text)
    if not objs:
        return None
    if expected_tool is not None:
        filt = [o for o in objs if o.get("tool") == expected_tool]
        if filt:
            objs = filt

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


def pick_obj_from_json(o):
    if not o:
        return None
    for k in ("best_obj", "tight_obj", "best_obj", "tight_obj"):
        v = o.get(k, None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                return None
    return None


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
            cases.append((case_id, mps))
    return cases


def main():
    root = ROOT
    cases_csv = root / "cases_with_symmetry.csv"
    if not cases_csv.exists():
        print(f"Missing {cases_csv}", file=sys.stderr)
        sys.exit(1)

    cases = load_symmetry_cases(cases_csv)
    if not cases:
        print("No symmetry cases with sym_ok=True found.", file=sys.stderr)
        sys.exit(1)

    # binaries / scripts to run (expected to be in the same folder)
    tools = [
        ("depth", [str(root / "orbit_branch_depth"), "--time-limit", str(TIME_LIMIT)]),
        ("sensing", [str(root / "orbit_branch_sensing"), "--time-limit", str(TIME_LIMIT)]),
        ("py", [sys.executable, str(root / "qp_bench.py")]),
        ("baseline", [str(root / "baseline"), str(TIME_LIMIT)]),
    ]

    # LD_LIBRARY_PATH: prefer local bliss if present
    env = os.environ.copy()
    local_bliss_lib = str((root / "local" / "bliss" / "lib").resolve())
    env["LD_LIBRARY_PATH"] = local_bliss_lib + ":" + env.get("LD_LIBRARY_PATH", "")

    out_csv = root / "benchmark_results.csv"
    fieldnames = ["test_case"]
    for name, _ in tools:
        fieldnames.append(f"{name}_time")
        fieldnames.append(f"{name}_obj")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for idx, (case_id, mps_path) in enumerate(cases, start=1):
            row = {"test_case": case_id}

            for name, cmd in tools:
                # C++ binaries expect the model file as the first argument: `prog model.mps [--time-limit ...]`
                if name in ("depth", "sensing"):
                    cmd_run = [cmd[0], mps_path, "--time-limit", str(TIME_LIMIT)]
                elif name == "baseline":
                    cmd_run = [str(root / "baseline"), mps_path, str(TIME_LIMIT)]
                elif name == "py":
                    cmd_run = [sys.executable, str(root / "qp_bench.py"), mps_path]
                else:
                    cmd_run = list(cmd)

                rc, out, to, elapsed = run_cmd(cmd_run, timeout_sec=TIME_LIMIT, env=env, cwd=str(root))
                j = extract_last_tool_json(out, expected_tool=None)
                obj = pick_obj_from_json(j)

                if rc == 124 or to:
                    # timed out
                    t = float(TIME_LIMIT)
                    if obj is None:
                        obj_val = "N/A"
                    else:
                        obj_val = obj
                else:
                    # completed (possibly with non-zero rc)
                    # prefer runtime reported in JSON if present
                    t = None
                    if j:
                        t = j.get("wall_runtime_sec") or j.get("tight_runtime_sec") or j.get("runtime")
                        try:
                            t = float(t) if t is not None else None
                        except Exception:
                            t = None
                    if t is None:
                        t = round(elapsed, 6)
                    obj_val = obj if obj is not None else "N/A"

                row[f"{name}_time"] = t
                row[f"{name}_obj"] = obj_val

            w.writerow(row)
            print(f"[{idx}/{len(cases)}] case {case_id} done")

    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
