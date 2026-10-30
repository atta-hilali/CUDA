#!/usr/bin/env python3
"""Save measured assignment outputs, metadata and a compact CSV table."""
import csv
import datetime
import json
import os
import platform
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def capture(command, optional=False):
    try:
        completed = subprocess.run(command, cwd=ROOT, text=True,
                                   capture_output=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        if optional:
            return str(exc)
        raise
    output = completed.stdout + completed.stderr
    if completed.returncode and not optional:
        raise RuntimeError(f"{' '.join(map(str, command))} failed:\n{output}")
    return output


def main():
    RESULTS.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata = {
        "timestamp_utc": timestamp,
        "commit": capture(["git", "rev-parse", "HEAD"], optional=True).strip(),
        "working_tree": capture(["git", "status", "--porcelain"], optional=True).strip(),
        "platform": platform.platform(),
        "gpu": capture(["nvidia-smi"], optional=True),
        "compiler": capture([os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc"), "--version"], optional=True),
        "timing_scope": "CUDA events, device execution; allocations and host/device copies excluded",
        "repeats": 100,
    }
    (RESULTS / f"metadata_{timestamp}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    jobs = [
        ("ger", ["3-ger/ger", str(m), str(n), "100"], {"m": m, "n": n, "batch": ""})
        for m, n in [(257, 129), (1024, 1024), (4000, 2000)]
    ] + [
        ("gemm", ["4-gemm/gemm", str(n), str(batch), "100"], {"m": n, "n": n, "batch": batch})
        for n in [4, 8, 16, 24, 32] for batch in [100, 1000, 10000]
    ]
    rows = []
    pattern = re.compile(r"^(Custom kernel|cuBLAS SGER|cuBLAS Strided Batched)\s+"
                         r"([0-9.eE+\-]+)\s+([0-9.eE+\-]+)\s+([0-9.eE+\-]+)$", re.M)
    # Write each raw result immediately, so an interrupted run retains completed work.
    for assignment, arguments, shape in jobs:
        output = capture([str(ROOT / arguments[0]), *arguments[1:]])
        if "TEST PASS" not in output:
            raise RuntimeError(f"Correctness failed for {arguments}:\n{output}")
        raw_path = RESULTS / f"{timestamp}_{assignment}_{'_'.join(arguments[1:])}.txt"
        raw_path.write_text("$ " + " ".join(arguments) + "\n" + output)
        parsed = pattern.findall(output)
        if len(parsed) != 2:
            raise RuntimeError(f"Expected two timing rows in {raw_path}")
        for implementation, time_us, gflops, error in parsed:
            rows.append({"assignment": assignment, **shape,
                         "implementation": implementation,
                         "time_us": time_us, "gflops": gflops, "error": error})
        print(assignment, shape, "PASS", flush=True)
    csv_path = RESULTS / f"benchmark_{timestamp}.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} measurements to {csv_path}")


if __name__ == "__main__":
    main()
