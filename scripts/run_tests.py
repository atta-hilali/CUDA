#!/usr/bin/env python3
"""Run the GPU correctness tests and both original assignment drivers."""
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(arguments, success=True):
    command = [str(ROOT / arguments[0]), *map(str, arguments[1:])]
    result = subprocess.run(command, text=True, capture_output=True, timeout=180)
    print("$", " ".join(map(str, arguments)))
    print(result.stdout, end="")
    print(result.stderr, end="")
    if success:
        if result.returncode != 0 or "FAIL" in result.stdout:
            raise SystemExit("Test failed")
        if "PASS" not in result.stdout:
            raise SystemExit("Test did not report a pass")
    elif result.returncode == 0 or "Usage:" not in result.stderr:
        raise SystemExit("Invalid arguments were not rejected")


def check_arguments():
    for executable, args in [
        ("3-ger/ger", [0, 8]), ("3-ger/ger", [8, -1]),
        ("3-ger/ger", [8, 8, 0]), ("3-ger/ger", ["8x", 8]),
        ("4-gemm/gemm", [0, 1]), ("4-gemm/gemm", [33, 1]),
        ("4-gemm/gemm", [16, 0]), ("4-gemm/gemm", [16, 1, 0]),
        ("4-gemm/gemm", ["abc", 1]),
    ]:
        run([executable, *args], success=False)
    print("All 9 invalid-argument checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arguments-only", action="store_true",
                        help="Check invalid CLI inputs without initializing CUDA")
    options = parser.parse_args()
    if options.arguments_only:
        check_arguments()
        return
    run(["tests/kernel_validation"])
    for m, n in [(1, 1), (31, 7), (33, 65), (257, 129), (4000, 2000)]:
        run(["3-ger/ger", m, n, 5])
    for n, batch in [(1, 1), (3, 7), (17, 33), (31, 100), (32, 1000)]:
        run(["4-gemm/gemm", n, batch, 5])
    check_arguments()
    print("All GPU kernel, driver and argument checks passed.")


if __name__ == "__main__":
    main()
