"""Answer-prior audit of the multi-image SAT-v2 training split.

Motivation: the Gemma LoRA arm collapses to the literal string "left" on BLINK
Multi-view Reasoning. This script measures whether that is explained by the
training corpus's answer distribution rather than by a broken run.

Writes analysis/results/sat_answer_prior.json.
"""
import collections, glob, json, os
import pyarrow.parquet as pq

CACHE = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub/datasets--array--SAT-v2")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "sat_answer_prior.json")

files = sorted(glob.glob(f"{CACHE}/**/*.parquet", recursive=True))
ans, qtype, n = collections.Counter(), collections.Counter(), 0
for f in files:
    for b in pq.ParquetFile(f).iter_batches(
            batch_size=2000, columns=["images", "correct_answer", "question_type"]):
        d = b.to_pydict()
        for i, imgs in enumerate(d["images"]):
            if isinstance(imgs, list) and len(imgs) >= 2:      # coupler-eligible rows only
                ans[str(d["correct_answer"][i]).strip().lower()] += 1
                qtype[str(d["question_type"][i])] += 1
                n += 1

lr = sum(v for k, v in ans.items() if "left" in k or "right" in k)
rep = {
    "cached_revision": "77d83da806d0a31f837be7b36d0dfbda230d8864",
    "multi_image_rows": n,
    "left_right_share_pct": round(100 * lr / n, 2) if n else None,
    "top_answers": [{"answer": k, "count": v, "pct": round(100 * v / n, 2)}
                    for k, v in ans.most_common(12)],
    "question_types": [{"type": k, "count": v, "pct": round(100 * v / n, 2)}
                       for k, v in qtype.most_common()],
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(rep, open(OUT, "w"), indent=2)
print(f"multi-image rows: {n}")
print(f"answers containing left/right: {rep['left_right_share_pct']}%")
for a in rep["top_answers"][:7]:
    print(f"   {a['answer'][:34]:36s} {a['pct']:5.2f}%")
