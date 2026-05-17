import argparse
import re
import statistics
from pathlib import Path

LAT_RE = re.compile(r"Action inference latency:\s*([0-9.]+)\s*ms")
SUMMARY_RE = re.compile(r"Action latency summary:\s*(\d+)\s*preds")
SUCCESS_RE = re.compile(r"Success:\s*(True|False)")
K_RE = re.compile(r"fixed_rec(\d+)(?:_steady)?\.log$")


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def fmt(x):
    return "-" if x is None else f"{x:.2f}"


def parse_log(path: Path, skip: int):
    text = path.read_text(errors="ignore")
    latencies = [float(m.group(1)) for m in LAT_RE.finditer(text)]

    success_match = None
    for m in SUCCESS_RE.finditer(text):
        success_match = m.group(1)

    k_match = K_RE.search(path.name)
    k = int(k_match.group(1)) if k_match else None

    steady = latencies[skip:] if len(latencies) > skip else []

    return {
        "file": path.name,
        "k": k,
        "success": success_match if success_match is not None else "-",
        "n": len(latencies),
        "raw_avg": statistics.mean(latencies) if latencies else None,
        "raw_median": statistics.median(latencies) if latencies else None,
        "raw_p95": pct(latencies, 95),
        "first": latencies[0] if len(latencies) >= 1 else None,
        "second": latencies[1] if len(latencies) >= 2 else None,
        "steady_n": len(steady),
        "steady_avg": statistics.mean(steady) if steady else None,
        "steady_median": statistics.median(steady) if steady else None,
        "steady_p90": pct(steady, 90),
        "steady_p95": pct(steady, 95),
        "steady_min": min(steady) if steady else None,
        "steady_max": max(steady) if steady else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="Log files, e.g. profiling/results/fixed_rec*.log")
    ap.add_argument("--skip", type=int, default=2, help="Number of initial action queries to exclude")
    args = ap.parse_args()

    rows = [parse_log(Path(p), args.skip) for p in args.logs]
    rows.sort(key=lambda r: (r["k"] is None, r["k"] if r["k"] is not None else 999999, r["file"]))

    headers = [
        "K", "success", "n", "first", "second",
        "raw_avg", "raw_p95",
        f"steady_avg(skip={args.skip})", "steady_median", "steady_p90", "steady_p95",
        "steady_min", "steady_max"
    ]

    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---:" if h not in {"success"} else "---" for h in headers]) + "|")

    for r in rows:
        vals = [
            str(r["k"]) if r["k"] is not None else r["file"],
            r["success"],
            str(r["n"]),
            fmt(r["first"]),
            fmt(r["second"]),
            fmt(r["raw_avg"]),
            fmt(r["raw_p95"]),
            fmt(r["steady_avg"]),
            fmt(r["steady_median"]),
            fmt(r["steady_p90"]),
            fmt(r["steady_p95"]),
            fmt(r["steady_min"]),
            fmt(r["steady_max"]),
        ]
        print("| " + " | ".join(vals) + " |")

    csv_path = Path("profiling/results/action_latency_steady_summary.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            vals = [
                str(r["k"]) if r["k"] is not None else r["file"],
                r["success"],
                str(r["n"]),
                fmt(r["first"]),
                fmt(r["second"]),
                fmt(r["raw_avg"]),
                fmt(r["raw_p95"]),
                fmt(r["steady_avg"]),
                fmt(r["steady_median"]),
                fmt(r["steady_p90"]),
                fmt(r["steady_p95"]),
                fmt(r["steady_min"]),
                fmt(r["steady_max"]),
            ]
            f.write(",".join(vals) + "\n")

    print(f"\nSaved CSV: {csv_path}")


if __name__ == "__main__":
    main()
