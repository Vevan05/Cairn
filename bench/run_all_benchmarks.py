"""Orchestrator: runs every benchmark and prints a consolidated report."""
from __future__ import annotations

import subprocess, sys, os, json


def run(script):
    print(f"\n===== {script} =====")
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    inp = subprocess.run(
        [sys.executable, script],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    print(inp.stdout)
    if inp.stderr:
        print("--- stderr ---")
        print(inp.stderr[:500])
    return inp.returncode


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    scripts = [
        "bench/benchmark_engine.py",
        "bench/benchmark_hnsw.py",
        "bench/benchmark_quantization.py",
        "bench/benchmark_day9.py",
    ]
    for s in scripts:
        run(s)
    print("\nAll benchmarks executed. Review the tables above.")
