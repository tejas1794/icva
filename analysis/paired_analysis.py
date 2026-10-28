"""Paired item-level analysis for the I-CVA evaluation matrix.

Produces, for every ordered pair of conditions:
  - accuracy difference, paired bootstrap 95% CI, discordant wins/losses,
    exact two-sided McNemar p
  - per-category breakdown
  - a sign test over categories (direction consistency)
and, separately, an answer-format compliance report.

Inference is item-level only; it quantifies evaluation uncertainty at a fixed
checkpoint, not training-seed variation.
"""
import glob
import json
import os
import pickle
import re
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.environ.get("PROJECT_ROOT", ".")  # directory holding outputs/ from scripts/
OUT = f"{ROOT}/outputs"
SEED = 20260722
NBOOT = 100_000

# condition -> (directory glob, filename stem)
GEMMA = OrderedDict([
    ("base",             (f"{OUT}/gemma4_base_authoritative/*",                "Gemma4")),
    ("independent_lora", (f"{OUT}/gemma40k_independent_lora_authoritative/*",  "Gemma4_LoRAOnly")),
    ("coupler_off",      (f"{OUT}/gemma40k_coupler_disabled/*",                "ICVAGemma4_40k_co_trained_LoRA_only")),
    ("joint_full",       (f"{OUT}/gemma40k_full_authoritative/*",              "ICVAGemma4_40k_with_LoRA")),
])
QWEN = OrderedDict([
    ("base",             (f"{OUT}/qwen3b_base_matched_authoritative/*",        "Qwen2_5_VL_3B_BaseMatched")),
    ("independent_lora", (f"{OUT}/qwen10k_independent_lora_authoritative/*",   "Qwen2_5_VL_3B_LoRAOnly")),
    ("coupler_off",      (f"{OUT}/qwen10k_coupler_disabled/*",                 "ICVAQwen2_5_VL_3B_co_trained_LoRA_only")),
    ("joint_full",       (f"{OUT}/qwen10k_full_authoritative/*",               "ICVAQwen2_5_VL_3B_with_LoRA")),
])
BENCH = {
    "BLINK": ("BLINK", "BLINK_exact_matching_result"),
    "MMSI":  ("MMSIBench_wo_circular", "MMSIBench_wo_circular_result"),
}
# BLINK categories with exactly one image; the coupler is bypassed on these.
SINGLE_IMAGE = {"Counting", "IQ_Test", "Object_Localization",
                "Relative_Depth", "Relative_Reflectance", "Spatial_Relation"}

OPTION_RX = re.compile(r"^\(?[A-D]\)?([.):\s].*)?$", re.S)


RESOLVED = {}


def first(pattern):
    """Resolve a result file under a condition directory.

    The evaluator leaves top-level convenience symlinks whose targets it later
    moves into `bak_*` folders, so a plain glob can match a dangling link.
    Search recursively for real files with the requested basename and choose
    deterministically: prefer paths outside `bak_*`, then the newest mtime.
    The chosen path is recorded so every reported number stays traceable.
    """
    root, name = pattern.rsplit("/*/", 1) if "/*/" in pattern else os.path.split(pattern)
    hits = [f for f in glob.glob(f"{root}/**/{name}", recursive=True)
            if os.path.isfile(f)]
    if not hits:
        raise FileNotFoundError(pattern)
    hits.sort(key=lambda f: ("bak_" in f, -os.path.getmtime(f), f))
    RESOLVED[pattern] = hits[0]
    return hits[0]


def hit_map(pkl_path):
    """Return {item_id: 0/1}. Handles both evaluator result-pickle layouts."""
    blob = pickle.load(open(pkl_path, "rb"))
    if isinstance(blob, dict) and "mcq_scored" in blob:      # MMSI layout
        scored = blob["mcq_scored"]
        return {str(i): int(round(float(h)))
                for i, h in zip(scored["index"], scored["hit"])}
    return {str(k): int(v["hit"]) for k, v in blob.items()}  # BLINK layout


def load(dir_glob, stem, bench_stem, pkl_stem):
    """Return DataFrame indexed by item id with correctness, category, prediction."""
    frame = pd.read_excel(first(f"{dir_glob}/{stem}_{bench_stem}.xlsx"))
    hits = hit_map(first(f"{dir_glob}/{stem}_{pkl_stem}.pkl"))
    ids = [str(i) for i in frame["index"]]
    missing = [i for i in ids if i not in hits]
    if missing:
        raise KeyError(f"{stem}: {len(missing)} ids absent from result pickle")
    return pd.DataFrame({
        "item": ids,
        "hit": [hits[i] for i in ids],
        "category": [str(c) for c in frame["category"]],
        "prediction": [str(p) for p in frame["prediction"]],
    }).set_index("item")


def paired(a, b, rng):
    """a, b are aligned 0/1 arrays: treatment and control."""
    n = len(a)
    delta = (a.mean() - b.mean()) * 100
    wins = int(((a == 1) & (b == 0)).sum())
    losses = int(((a == 0) & (b == 1)).sum())
    if wins + losses:
        p = float(stats.binomtest(wins, wins + losses, 0.5).pvalue)
    else:
        p = 1.0
    diff = a - b
    if np.all(diff == 0):
        lo = hi = 0.0
    else:
        idx = rng.integers(0, n, size=(NBOOT, n))
        boot = diff[idx].mean(axis=1) * 100
        lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(n=n, treat=a.mean() * 100, ctrl=b.mean() * 100, delta=delta,
                lo=lo, hi=hi, wins=wins, losses=losses, p=p)


def populations(df):
    """Named item subsets within BLINK."""
    single = df["category"].isin(SINGLE_IMAGE)
    return {
        "all": df.index,
        "multi_view_reasoning": df.index[df["category"] == "Multi-view_Reasoning"],
        "single_image_retention": df.index[single],
        "multi_image_nontarget": df.index[~single & (df["category"] != "Multi-view_Reasoning")],
    }


def compliance(series):
    return np.array([bool(OPTION_RX.match(s.strip())) for s in series])


def run(model_name, conditions):
    rng = np.random.default_rng(SEED)
    data = {}
    for cond, (dir_glob, stem) in conditions.items():
        for bench, (bench_stem, pkl_stem) in BENCH.items():
            data[(cond, bench)] = load(dir_glob, stem, bench_stem, pkl_stem)

    # align every condition on a common item order per benchmark
    for bench in BENCH:
        ref = data[("base", bench)].index
        for cond in conditions:
            d = data[(cond, bench)]
            assert set(d.index) == set(ref), f"{model_name}/{cond}/{bench}: item mismatch"
            data[(cond, bench)] = d.loc[ref]
        cats = data[("base", bench)]["category"]
        for cond in conditions:
            assert (data[(cond, bench)]["category"] == cats).all(), \
                f"{model_name}/{cond}/{bench}: category mismatch"

    lines = [f"# {model_name}", ""]
    report = {"model": model_name, "contrasts": {}, "sign_tests": {}, "compliance": {}}

    contrasts = [("joint_full", "base"), ("joint_full", "independent_lora"),
                 ("joint_full", "coupler_off"), ("independent_lora", "base"),
                 ("coupler_off", "base")]

    for treat, ctrl in contrasts:
        key = f"{treat}_vs_{ctrl}"
        lines += [f"## {key}", "",
                  "| Population | n | Treat | Ctrl | Delta (pp) | 95% CI | W/L | McNemar p |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        rows = {}
        blink = data[(treat, "BLINK")]
        pops = populations(blink)
        for name, idx in pops.items():
            r = paired(data[(treat, "BLINK")].loc[idx, "hit"].values,
                       data[(ctrl, "BLINK")].loc[idx, "hit"].values, rng)
            rows[f"blink_{name}"] = r
        r = paired(data[(treat, "MMSI")]["hit"].values,
                   data[(ctrl, "MMSI")]["hit"].values, rng)
        rows["mmsi_all"] = r
        # predeclared pooled cross-view endpoint: MMSI + BLINK MVR
        mvr = pops["multi_view_reasoning"]
        pooled_t = np.concatenate([data[(treat, "MMSI")]["hit"].values,
                                   data[(treat, "BLINK")].loc[mvr, "hit"].values])
        pooled_c = np.concatenate([data[(ctrl, "MMSI")]["hit"].values,
                                   data[(ctrl, "BLINK")].loc[mvr, "hit"].values])
        rows["pooled_cross_view"] = paired(pooled_t, pooled_c, rng)

        for name, r in rows.items():
            lines.append(
                f"| {name} | {r['n']} | {r['treat']:.2f} | {r['ctrl']:.2f} | "
                f"{r['delta']:+.2f} | [{r['lo']:+.2f}, {r['hi']:+.2f}] | "
                f"{r['wins']}/{r['losses']} | {r['p']:.4f} |")
        lines.append("")
        report["contrasts"][key] = rows

        # per-category table + sign test across categories
        cat_rows = []
        for bench in BENCH:
            t, c = data[(treat, bench)], data[(ctrl, bench)]
            for cat, grp in t.groupby("category"):
                idx = grp.index
                a, b = t.loc[idx, "hit"].values, c.loc[idx, "hit"].values
                cat_rows.append((bench, cat, len(idx),
                                 a.mean() * 100, b.mean() * 100,
                                 (a.mean() - b.mean()) * 100))
        deltas = np.array([r[5] for r in cat_rows])
        up = int((deltas > 0).sum())
        down = int((deltas < 0).sum())
        tie = int((deltas == 0).sum())
        sp = float(stats.binomtest(up, up + down, 0.5).pvalue) if up + down else 1.0
        report["sign_tests"][key] = dict(up=up, down=down, tie=tie, p=sp,
                                         n_categories=len(cat_rows))
        report.setdefault("categories", {})[key] = [
            dict(bench=b, category=c, n=n, treat=tv, ctrl=cv, delta=d)
            for b, c, n, tv, cv, d in cat_rows]
        lines += [f"**Sign test over {len(cat_rows)} categories:** "
                  f"{up} up, {down} down, {tie} tied, binomial p = {sp:.5f}", "",
                  "| Bench | Category | n | Treat | Ctrl | Delta |",
                  "|---|---|---:|---:|---:|---:|"]
        for bench, cat, n, tv, cv, d in cat_rows:
            lines.append(f"| {bench} | {cat} | {n} | {tv:.2f} | {cv:.2f} | {d:+.2f} |")
        lines.append("")

    # answer-format compliance on BLINK
    lines += ["## Answer-format compliance (BLINK)", "",
              "| Condition | Compliant % | Acc | Acc on compliant | Acc on non-compliant |",
              "|---|---:|---:|---:|---:|"]
    for cond in conditions:
        d = data[(cond, "BLINK")]
        c = compliance(d["prediction"])
        nc = int((~c).sum())
        rec = dict(compliant_pct=float(c.mean() * 100), acc=float(d["hit"].mean() * 100),
                   acc_compliant=float(d["hit"].values[c].mean() * 100),
                   acc_noncompliant=float(d["hit"].values[~c].mean() * 100) if nc else None,
                   n_noncompliant=nc)
        report["compliance"][cond] = rec
        anc = f"{rec['acc_noncompliant']:.2f}" if nc else "n/a"
        lines.append(f"| {cond} | {rec['compliant_pct']:.2f} | {rec['acc']:.2f} | "
                     f"{rec['acc_compliant']:.2f} | {anc} (n={nc}) |")
    lines.append("")

    # negative transfer restricted to items where both conditions parsed cleanly
    base_c = compliance(data[("base", "BLINK")]["prediction"])
    lora_c = compliance(data[("independent_lora", "BLINK")]["prediction"])
    both = base_c & lora_c
    b = data[("base", "BLINK")]["hit"].values[both]
    l = data[("independent_lora", "BLINK")]["hit"].values[both]
    lines += [f"**Format-controlled negative transfer:** on the {int(both.sum())} BLINK items "
              f"where base and independent LoRA both emit a parseable option, "
              f"base = {b.mean()*100:.2f}, independent LoRA = {l.mean()*100:.2f} "
              f"({(l.mean()-b.mean())*100:+.2f} pp).", ""]
    report["format_controlled_negative_transfer"] = dict(
        n=int(both.sum()), base=float(b.mean() * 100), lora=float(l.mean() * 100),
        delta=float((l.mean() - b.mean()) * 100))
    return "\n".join(lines), report


if __name__ == "__main__":
    md, blob = [], {}
    for name, conds in [("Gemma-4-E4B-it", GEMMA), ("Qwen2.5-VL-3B", QWEN)]:
        text, rep = run(name, conds)
        md.append(text)
        blob[name] = rep
    os.makedirs(f"{ROOT}/analysis/results", exist_ok=True)
    header = (f"# Paired item-level analysis\n\n"
              f"Paired bootstrap: {NBOOT:,} replicates, seed {SEED}. "
              f"McNemar tests are exact two-sided binomial tests on discordant pairs. "
              f"Category contrasts are descriptive and not multiplicity-adjusted; "
              f"the sign test over categories is the only category-level inference.\n\n")
    open(f"{ROOT}/analysis/results/paired_analysis.md", "w").write(header + "\n".join(md))
    blob["_resolved_source_files"] = {k: v.replace(ROOT + "/", "")
                                      for k, v in sorted(RESOLVED.items())}
    json.dump(blob, open(f"{ROOT}/analysis/results/paired_analysis.json", "w"), indent=2)
    print("wrote analysis/results/paired_analysis.{md,json}")
