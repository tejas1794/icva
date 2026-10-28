#!/usr/bin/env python3
"""Recompute headline numbers in the paper directly from the shipped per-question logs.

Usage:  python verify.py
Requires: pandas, numpy, openpyxl  (no GPU, no model weights)
"""
import glob, os, sys
import numpy as np, pandas as pd
from math import comb

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
SINGLE = {"Counting","IQ_Test","Object_Localization",
          "Relative_Depth","Relative_Reflectance","Spatial_Relation"}
rng = np.random.default_rng(20260722)

def load(arm, bench="BLINK"):
    hits = [f for f in glob.glob(os.path.join(LOGS, arm, "*.xlsx")) if bench in os.path.basename(f)]
    if not hits:
        return None
    df = pd.read_excel(hits[0]).set_index("index").sort_index()
    df["hit"] = df["hit"].astype(float).round().astype(int)
    return df

def paired(a, b, mask=None, reps=100_000):
    """a minus b on matched items; returns delta, CI, wins, losses."""
    i = a.index.intersection(b.index)
    a, b = a.loc[i], b.loc[i]
    if mask is not None:
        m = mask(b)
        a, b = a[m], b[m]
    ah, bh = a["hit"].values, b["hit"].values
    d = (ah - bh).astype(float)
    bt = d[rng.integers(0, len(d), (reps, len(d)))].mean(axis=1) * 100
    wins = int(((bh == 0) & (ah == 1)).sum())
    losses = int(((bh == 1) & (ah == 0)).sum())
    return d.mean()*100, np.percentile(bt,2.5), np.percentile(bt,97.5), wins, losses, len(d)

def show(label, res, expect=None):
    if res is None:
        print(f"  {label:<44} [logs not present]"); return
    d, lo, hi, w, l, n = res
    tag = ""
    if expect is not None:
        tag = "  OK" if abs(d - expect) < 0.02 else f"  <-- paper says {expect:+.2f}"
    print(f"  {label:<44} n={n:>5}  {d:+6.2f}  CI[{lo:+.2f},{hi:+.2f}]  W/L={w}/{l}{tag}")

print("=" * 88)
print("I-CVA -- verification of reported numbers from per-question logs")
print("=" * 88)

print("\nSection 4.1  Negative transfer under LoRA (Gemma-4, BLINK)")
gb, gl = load("gemma_base"), load("gemma_lora_only")
if gb is not None and gl is not None:
    show("LoRA - base, all BLINK", paired(gl, gb), -9.26)
    show("LoRA - base, Multi-view Reasoning",
         paired(gl, gb, lambda d: d["category"] == "Multi-view_Reasoning"))

print("\nSection 4.2  Convex gate vs base (Gemma-4, BLINK)")
gg = load("gemma_gate")
if gb is not None and gg is not None:
    show("gate - base, all BLINK", paired(gg, gb), -6.79)
    show("gate - base, Multi-view Reasoning",
         paired(gg, gb, lambda d: d["category"] == "Multi-view_Reasoning"), +6.02)
    show("gate - base, Visual Correspondence",
         paired(gg, gb, lambda d: d["category"] == "Visual_Correspondence"), -43.02)
    show("gate - base, six bypassed categories",
         paired(gg, gb, lambda d: d["category"].isin(SINGLE)), 0.00)

print("\nSection 4.3  I-CVA vs base on the predeclared BLINK populations")
for bk, base_arm, icva_arm in [("Qwen", "qwen_base", "qwen_icva"),
                               ("Gemma", "gemma_base", "gemma_icva")]:
    b, a = load(base_arm), load(icva_arm)
    if b is None or a is None:
        print(f"  {bk}: logs not present"); continue
    show(f"{bk}: target (Multi-view Reasoning)",
         paired(a, b, lambda d: d["category"] == "Multi-view_Reasoning"), +2.26)
    show(f"{bk}: retention (six single-image cats)",
         paired(a, b, lambda d: d["category"].isin(SINGLE)), 0.00)
    show(f"{bk}: non-target multi-image",
         paired(a, b, lambda d: ~d["category"].isin(SINGLE) & (d["category"] != "Multi-view_Reasoning")))

print("\nSection 4.4  Avoided damage on the second backbone (Qwen)")
for bench, exp_lora, exp_icva in [("MindCube", -4.33, +3.94),
                                  ("EmbSpatial", None, -0.08),
                                  ("ViewSpatial", None, -0.05)]:
    b = load("qwen_base", bench); l = load("qwen_lora_only", bench); a = load("qwen_icva", bench)
    if b is None or l is None or a is None:
        print(f"  {bench}: logs not present"); continue
    show(f"{bench}: LoRA - base", paired(l, b), exp_lora)
    show(f"{bench}: I-CVA - LoRA", paired(a, l), exp_icva)
    show(f"{bench}: I-CVA - base", paired(a, b))

print("\nSection 4.6  Matched control: coupler enabled vs bypassed, weights fixed")
on, off = load("qwen_cotrained_on"), load("qwen_cotrained_off")
if on is not None and off is not None:
    show("Qwen: single-image items (expect exactly 0)",
         paired(on, off, lambda d: d["category"].isin(SINGLE)), 0.00)
    # the reported population pools BLINK Multi-view Reasoning with all of MMSI-Bench
    mon, moff = load("qwen_cotrained_on", "MMSI"), load("qwen_cotrained_off", "MMSI")
    if mon is not None and moff is not None:
        bi = on.index.intersection(off.index)
        bm = on.loc[bi][on.loc[bi]["category"] == "Multi-view_Reasoning"]
        bo = off.loc[bm.index]
        mi = mon.index.intersection(moff.index)
        ah = np.concatenate([bm["hit"].values, mon.loc[mi]["hit"].values])
        bh = np.concatenate([bo["hit"].values, moff.loc[mi]["hit"].values])
        d = (ah - bh).astype(float)
        bt = d[rng.integers(0, len(d), (100_000, len(d)))].mean(axis=1) * 100
        w = int(((bh == 0) & (ah == 1)).sum()); l = int(((bh == 1) & (ah == 0)).sum())
        show("Qwen: pooled MVR + MMSI (reported population)",
             (d.mean()*100, np.percentile(bt,2.5), np.percentile(bt,97.5), w, l, len(d)), +0.44)

print("\nSection 4.7  Refinement depth (Qwen, BLINK)")
qb = load("qwen_base")
ks = sorted(glob.glob(os.path.join(LOGS, "qwen_ksweep", "*BLINK*.xlsx")))
if qb is not None and ks:
    for f in ks:
        k = os.path.basename(f).split("_K")[-1][0] if "_K" in os.path.basename(f) else "?"
        df = pd.read_excel(f).set_index("index").sort_index()
        df["hit"] = df["hit"].astype(float).round().astype(int)
        show(f"K={k}: Multi-view Reasoning vs base",
             paired(df, qb, lambda d: d["category"] == "Multi-view_Reasoning"))

print("\n" + "=" * 88)
print("Every number above is derived only from the shipped logs. No weights required.")
