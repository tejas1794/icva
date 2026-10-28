"""Paired coupler-on vs coupler-off analysis for the Gemma-40k matched suite.

Reads whatever benchmark pairs have finished and reports, per benchmark,
the accuracy difference with paired bootstrap CI, discordant wins/losses,
and an exact McNemar test. Also emits a sign test over the per-benchmark
category deltas. Re-runnable at any time; incomplete pairs are skipped.
"""
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.environ.get("PROJECT_ROOT", ".")  # directory holding outputs/ from scripts/
SUITE = f"{ROOT}/outputs/gemma40k_matched_suite_20260826"
ON_STEM = "ICVAGemma4_40k_with_LoRA"
OFF_STEM = "ICVAGemma4_40k_co_trained_LoRA_only"
BENCHES = ["MindCubeBench_tiny_raw_qa", "EmbSpatialBench", "ViewSpatialBench"]
SEED = 20260722
NBOOT = 100_000


def hit_map(path):
    blob = pickle.load(open(path, "rb"))
    if isinstance(blob, dict) and "mcq_scored" in blob:
        s = blob["mcq_scored"]
        return {str(i): int(round(float(h))) for i, h in zip(s["index"], s["hit"])}
    return {str(k): int(v["hit"]) for k, v in blob.items()}


def real_files(root, name):
    hits = [f for f in glob.glob(f"{root}/**/{name}", recursive=True)
            if os.path.isfile(f)]
    hits.sort(key=lambda f: ("bak_" in f, -os.path.getmtime(f), f))
    return hits


def load_arm(arm_dir, stem, bench):
    xs = real_files(arm_dir, f"{stem}_{bench}.xlsx")
    ps = real_files(arm_dir, f"{stem}_{bench}_result.pkl")
    if not xs or not ps:
        return None
    frame = pd.read_excel(xs[0])
    hits = hit_map(ps[0])
    ids = [str(i) for i in frame["index"]]
    if any(i not in hits for i in ids):
        return None
    cat_col = "category" if "category" in frame.columns else None
    return pd.DataFrame({
        "item": ids,
        "hit": [hits[i] for i in ids],
        "category": [str(frame[cat_col][k]) for k in range(len(ids))] if cat_col
                    else ["all"] * len(ids),
    }).set_index("item"), os.path.relpath(xs[0], ROOT)


def paired(a, b, rng):
    n = len(a)
    wins = int(((a == 1) & (b == 0)).sum())
    losses = int(((a == 0) & (b == 1)).sum())
    p = float(stats.binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0
    diff = a - b
    if np.all(diff == 0):
        lo = hi = 0.0
    else:
        idx = rng.integers(0, n, size=(NBOOT, n))
        boot = diff[idx].mean(axis=1) * 100
        lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(n=n, on=a.mean() * 100, off=b.mean() * 100,
                delta=(a.mean() - b.mean()) * 100, lo=lo, hi=hi,
                wins=wins, losses=losses, p=p)


def main():
    rng = np.random.default_rng(SEED)
    lines = ["# Gemma-4-E4B-it 40k: matched coupler-on vs coupler-off", "",
             f"Paired bootstrap {NBOOT:,} replicates, seed {SEED}; exact two-sided McNemar. "
             "Both arms share every weight; only adapter execution differs.", "",
             "| Benchmark | n | On | Off | Delta (pp) | 95% CI | W/L | p |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    report, cat_lines, sources = {}, [], {}
    for bench in BENCHES:
        on = load_arm(f"{SUITE}/on_{bench}", ON_STEM, bench)
        off = load_arm(f"{SUITE}/off_{bench}", OFF_STEM, bench)
        if on is None or off is None:
            lines.append(f"| {bench} | _pending_ | | | | | | |")
            continue
        (a_df, a_src), (b_df, b_src) = on, off
        if set(a_df.index) != set(b_df.index):
            lines.append(f"| {bench} | _item mismatch_ | | | | | | |")
            continue
        b_df = b_df.loc[a_df.index]
        r = paired(a_df["hit"].values, b_df["hit"].values, rng)
        report[bench] = r
        sources[bench] = {"on": a_src, "off": b_src}
        lines.append(f"| {bench} | {r['n']} | {r['on']:.2f} | {r['off']:.2f} | "
                     f"{r['delta']:+.2f} | [{r['lo']:+.2f}, {r['hi']:+.2f}] | "
                     f"{r['wins']}/{r['losses']} | {r['p']:.4f} |")
        cat_lines += [f"", f"### {bench} categories", "",
                      "| Category | n | On | Off | Delta |", "|---|---:|---:|---:|---:|"]
        for cat, grp in a_df.groupby("category"):
            i = grp.index
            av, bv = a_df.loc[i, "hit"].values, b_df.loc[i, "hit"].values
            cat_lines.append(f"| {cat} | {len(i)} | {av.mean()*100:.2f} | "
                             f"{bv.mean()*100:.2f} | {(av.mean()-bv.mean())*100:+.2f} |")
    lines += cat_lines + [""]
    os.makedirs(f"{ROOT}/analysis/results", exist_ok=True)
    open(f"{ROOT}/analysis/results/gemma40k_matched_suite.md", "w").write("\n".join(lines))
    json.dump({"pairs": report, "sources": sources},
              open(f"{ROOT}/analysis/results/gemma40k_matched_suite.json", "w"), indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
