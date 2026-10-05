"""Compare recall, QPS and build times between a git ref and the working tree.

Builds the ref's Python module in a temporary copy, then runs each side's own
bench_indexes.py alternately (base, current, base, current, ...) on this
machine. QPS is the best and build time the fastest across rounds, so noise and
slow drift hit both sides equally. Wall-clock numbers are only meaningful on an
idle machine.

Run from the repo root after ./build.sh:
    python3 benchmarks/compare.py                          # HEAD vs working tree, siftsmall
    python3 benchmarks/compare.py main --rounds 3
    python3 benchmarks/compare.py HEAD~1 --dataset sift --rounds 1
"""

import argparse
import csv
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEED = 1


def sh(cmd, **kw):
    subprocess.run(cmd, check=True, **kw)


def build_base(ref: str, root: Path):
    archive = subprocess.run(["git", "archive", ref], cwd=REPO, check=True, capture_output=True).stdout
    sh(["tar", "-x", "-C", str(root)], input=archive)
    (root / "benchmarks" / "data").symlink_to(REPO / "benchmarks" / "data")

    # Reuse already-downloaded dependency sources instead of fetching them again.
    defines = ["-DCMAKE_BUILD_TYPE=Release", "-DVECENGINE_BUILD_TESTS=OFF"]
    for name in ("catch2", "benchmark", "nanobind"):
        src = REPO / "build" / "_deps" / f"{name}-src"
        if src.is_dir():
            defines.append(f"-DFETCHCONTENT_SOURCE_DIR_{name.upper()}={src}")
    print(f"building {ref} in {root}")
    sh(["cmake", "-S", str(root), "-B", str(root / "build"), *defines], stdout=subprocess.DEVNULL)
    sh(["cmake", "--build", str(root / "build"), "--target", "vecengine_py", "-j"], stdout=subprocess.DEVNULL)


def run_bench(root: Path, dataset: str, out_dir: Path) -> list[dict]:
    # Older bench scripts may lack --seed / --out-dir; they write to their own results dir.
    script = root / "benchmarks" / "bench_indexes.py"
    text = script.read_text()
    cmd = [sys.executable, str(script), "--dataset", dataset]
    if "--seed" in text:
        cmd += ["--seed", str(SEED)]
    if "--out-dir" in text:
        cmd += ["--out-dir", str(out_dir)]
    else:
        out_dir = root / "benchmarks" / "results"
    env = dict(os.environ, PYTHONPATH=str(root / "build"))
    sh(cmd, cwd=root, env=env, stdout=subprocess.DEVNULL)
    with open(out_dir / f"{dataset}.csv") as f:
        return list(csv.DictReader(f))


def merge(runs: list[list[dict]]) -> dict:
    merged = {}
    for rows in runs:
        for r in rows:
            key = (r["index"], r["params"])
            m = merged.setdefault(key, dict(recall=float(r["recall"]), qps=0.0,
                                            train_s=math.inf, add_s=math.inf))
            m["qps"] = max(m["qps"], float(r["qps"]))
            m["train_s"] = min(m["train_s"], float(r["train_s"]))
            m["add_s"] = min(m["add_s"], float(r["add_s"]))
    return merged


def report(base: dict, cur: dict, ref: str):
    print(f"\n{'index':<18} {'params':<12} {'recall':>15} {'qps ' + ref:>14} {'qps current':>12} {'ratio':>7}")
    ratios = []
    for key in cur:
        if key not in base:
            continue
        b, c = base[key], cur[key]
        ratio = c["qps"] / b["qps"]
        ratios.append(ratio)
        flag = "  <-- slower" if ratio < 0.9 else ""
        recall = f"{b['recall']:.3f}->{c['recall']:.3f}"
        print(f"{key[0]:<18} {key[1]:<12} {recall:>15} {b['qps']:>14.0f} {c['qps']:>12.0f} {ratio:>6.2f}x{flag}")

    print(f"\n{'build':<18} {'train ' + ref:>14} {'train current':>14} {'add ' + ref:>12} {'add current':>12}")
    seen = set()
    for (index, _), c in cur.items():
        name = index.split("-ADC")[0].split("-SDC")[0]
        if name in seen:
            continue
        seen.add(name)
        b = next((v for (i, _), v in base.items() if i == index), None)
        if b is not None:
            print(f"{name:<18} {b['train_s']:>13.2f}s {c['train_s']:>13.2f}s {b['add_s']:>11.2f}s {c['add_s']:>11.2f}s")

    if ratios:
        geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        print(f"\nQPS geometric mean ratio over {len(ratios)} rows: {geo:.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ref", nargs="?", default="HEAD", help="git ref to compare against (default HEAD)")
    parser.add_argument("--dataset", default="siftsmall", help="siftsmall or sift")
    parser.add_argument("--rounds", type=int, default=2, help="alternating runs per side")
    args = parser.parse_args()

    if not (REPO / "build" / "CMakeCache.txt").exists():
        sys.exit("run ./build.sh first: the working tree is benchmarked from build/")
    print("building working tree")
    sh(["cmake", "--build", str(REPO / "build"), "--target", "vecengine_py", "-j"], stdout=subprocess.DEVNULL)

    with tempfile.TemporaryDirectory(prefix="vecengine-compare-") as tmp:
        tmp = Path(tmp)
        base_root = tmp / "base"
        base_root.mkdir()
        build_base(args.ref, base_root)

        base_runs, cur_runs = [], []
        for i in range(args.rounds):
            print(f"round {i + 1}/{args.rounds}: {args.ref}")
            base_runs.append(run_bench(base_root, args.dataset, tmp / f"base-{i}"))
            print(f"round {i + 1}/{args.rounds}: current")
            cur_runs.append(run_bench(REPO, args.dataset, tmp / f"cur-{i}"))

        report(merge(base_runs), merge(cur_runs), args.ref)


if __name__ == "__main__":
    main()
