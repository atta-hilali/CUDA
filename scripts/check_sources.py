#!/usr/bin/env python3
"""Validate source completeness and syntax without importing Colab dependencies."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for path in [*ROOT.glob("scripts/*.py"), ROOT / "tests/host_check.py"]:
        compile(path.read_text(), str(path), "exec")
    notebook = json.loads((ROOT / "CUDA_HW4_Colab.ipynb").read_text())
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook cell {index}", "exec")
    for relative in ["3-ger/ger.cu", "4-gemm/gemm.cu"]:
        source = (ROOT / relative).read_text()
        if "TODO" in source or "int main(" not in source:
            raise RuntimeError(f"Incomplete or moved assignment driver: {relative}")
    print("Python scripts, notebook syntax and assignment structure: PASS")


if __name__ == "__main__":
    main()
