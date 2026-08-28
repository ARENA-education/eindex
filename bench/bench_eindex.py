"""Benchmark: compile-once `eindex` vs the original implementation vs hand-written torch, swept over
batch size for several patterns. Prints a table and saves `bench/bench_eindex_{cpu,cuda}.png`
(git-ignored -- paste them into a PR, don't commit them).

    pip install -e ".[test]" matplotlib
    python bench/bench_eindex.py

Methods (per pattern, per size):
  - original eindex 0.1.1      : re-parses + re-validates the pattern on every call (tests/reference)
  - eindex (drop-in, cached)   : this package's `eindex(...)`; compiled once per pattern, lru_cache'd
  - compile_eindex closure     : the pre-compiled closure, called directly
  - native torch               : the best hand-written gather / advanced index
"""

import sys
import time
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))  # -> `reference` (the vendored original)
from reference import eindex as ref_eindex  # noqa: E402

from eindex import compile_eindex, eindex  # noqa: E402

warnings.filterwarnings("ignore")
torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = DEVICE


def bench(fn, reps):
    cuda = DEVICE.type == "cuda"
    fn()  # warmup / compile / cache
    if cuda:
        torch.cuda.synchronize()
    best = float("inf")
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        if cuda:
            torch.cuda.synchronize()  # GPU is async; sync before stopping the clock
        best = min(best, (time.perf_counter() - t) / reps)
    return best * 1e6  # microseconds per call


CONFIGS = [
    (
        "1-D gather:  'batch [batch]'",
        "batch [batch]",
        lambda n: (torch.randint(0, 7, (n, 7), device=D), [torch.randint(0, 7, (n,), device=D)]),
        lambda arr, idx: arr.gather(1, idx[0].unsqueeze(1)).squeeze(1),
    ),
    (
        "2-D gather:  'batch seq [batch seq]'",
        "batch seq [batch seq]",
        lambda n: (torch.randn(n, 16, 32, device=D), [torch.randint(0, 32, (n, 16), device=D)]),
        lambda arr, idx: arr.gather(2, idx[0].unsqueeze(2)).squeeze(2),
    ),
    (
        "multi-index:  '... [b s] [b s]'",
        "batch seq [batch seq] [batch seq]",
        lambda n: (
            torch.randn(n, 16, 32, 16, device=D),
            [torch.randint(0, 32, (n, 16), device=D), torch.randint(0, 16, (n, 16), device=D)],
        ),
        lambda arr, idx: arr[torch.arange(arr.shape[0], device=D)[:, None], torch.arange(16, device=D), idx[0], idx[1]],
    ),
    (
        "offset:  'batch seq [batch seq+1]'",
        "batch seq [batch seq+1]",
        lambda n: (torch.randn(n, 16, 32, device=D), [torch.randint(0, 32, (n, 16), device=D)]),
        lambda arr, idx: arr[:, :-1].gather(2, idx[0][:, 1:].unsqueeze(2)).squeeze(2),
    ),
]
SIZES = [64, 256, 1024, 4096, 16384, 65536, 262144]
METHODS = ["original eindex 0.1.1", "eindex (drop-in, cached)", "compile_eindex closure", "native torch"]
COLORS = dict(zip(METHODS, ["#d62728", "#1f77b4", "#2ca02c", "#7f7f7f"]))


def main():
    results = {}
    for title, pat, make, native in CONFIGS:
        f = compile_eindex(pat)
        res = {k: [] for k in METHODS}
        for n in SIZES:
            arr, idx = make(n)
            assert torch.equal(eindex(arr, *idx, pat), ref_eindex(arr, *idx, pat))
            reps = max(8, min(200, int(2e5 / n)))
            res[METHODS[0]].append(bench(lambda: ref_eindex(arr, *idx, pat), reps))
            res[METHODS[1]].append(bench(lambda: eindex(arr, *idx, pat), reps))
            res[METHODS[2]].append(bench(lambda: f(arr, *idx), reps))
            res[METHODS[3]].append(bench(lambda: native(arr, idx), reps))
        results[title] = res
        print(f"\n{title}   [{DEVICE.type}]  (us/call)")
        print(f"  {'N':>8s} | " + " | ".join(f"{m:>24s}" for m in METHODS))
        for i, n in enumerate(SIZES):
            print(f"  {n:>8d} | " + " | ".join(f"{res[m][i]:24.1f}" for m in METHODS))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping plot)")
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (title, res) in zip(axes.flat, results.items()):
        for m, ys in res.items():
            ax.plot(SIZES, ys, "o-", label=m, color=COLORS[m], lw=1.8, ms=4)
        sp = max(r / c for r, c in zip(res[METHODS[0]], res[METHODS[2]]))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{title}\n(compile-once up to {sp:.0f}x faster than the original)", fontsize=9)
        ax.set_xlabel("batch size N (log)")
        ax.set_ylabel("us / call (log)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7.5)
    fig.suptitle(f"eindex runtime vs size ({DEVICE.type.upper()}) -- lower is better", fontsize=13)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / f"bench_eindex_{DEVICE.type}.png"
    fig.savefig(out, dpi=130)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
