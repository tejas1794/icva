"""SAT-v2 training-distribution accuracy for the Gemma arms.

Answers a narrow question: did the LoRA recipe fit its own training objective, or
did it fail to fit at all? Prompt construction mirrors train_v3.build_messages and
the generation path mirrors the trainer's validation loop, so the numbers are
comparable to what training optimised.

Note: the training stream was not pinned to a dataset revision, so these rows are
drawn from the same distribution but cannot be certified disjoint from training.
They measure fit, not generalisation.
"""
import argparse, collections, glob, io, json, os, sys
import pyarrow.parquet as pq
import torch
from PIL import Image

CACHE = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub/datasets--array--SAT-v2")
CKPT_ROOT = os.environ.get("CKPT_ROOT", "checkpoints")  # see README, "Model weights"

def build_messages(sample, num_images):
    opts = sample.get("answers") or []
    system = ("You are a spatial reasoning engine. Answer the multiple-choice question "
              "with the exact text of the correct option and nothing else.")
    q = f"{sample['question']}\nOptions:\n" + "\n".join(opts)
    return [{"role": "user", "content": [{"type": "text", "text": system},
                                         *[{"type": "image"} for _ in range(num_images)],
                                         {"type": "text", "text": q}]}]

def load_rows(n, skip=0, stride=1):
    """Iterate in the same forward shard order the trainer consumed, skipping the
    first `skip` eligible rows so the remainder was never seen at `skip` steps.
    `stride` spreads the sample across the whole unseen region rather than taking
    one contiguous block, which would sample a single question template."""
    rows, seen, kept = [], 0, 0
    for f in sorted(glob.glob(f"{CACHE}/**/*.parquet", recursive=True)):
        for b in pq.ParquetFile(f).iter_batches(
                batch_size=200, columns=["images", "question", "answers", "correct_answer"]):
            d = b.to_pydict()
            for i, imgs in enumerate(d["images"]):
                if isinstance(imgs, list) and len(imgs) >= 2:
                    seen += 1
                    if seen <= skip:
                        continue
                    kept += 1
                    if (kept - 1) % stride:
                        continue
                    rows.append({"images": imgs[:4], "question": d["question"][i],
                                 "answers": d["answers"][i], "correct_answer": d["correct_answer"][i]})
                    if len(rows) >= n:
                        return rows
    return rows


def to_pil(x):
    if isinstance(x, dict) and "bytes" in x: return Image.open(io.BytesIO(x["bytes"])).convert("RGB")
    if isinstance(x, (bytes, bytearray)):    return Image.open(io.BytesIO(x)).convert("RGB")
    raise TypeError(type(x))


def _register_hook(model, coupler):
    """Verbatim copy of _register_icva_qwen_hook from the evaluator wrapper."""
    def coupler_hook(module, args_in, kwargs_in, output):
        grid_thw = kwargs_in.get("grid_thw")
        if grid_thw is None and len(args_in) > 1:
            grid_thw = args_in[1]
        if grid_thw is None:
            return output
        merge = module.spatial_merge_size
        split_sizes = (grid_thw.prod(-1) // merge**2).tolist()
        if len(split_sizes) < 2:
            return output
        image_embeds = list(torch.split(output.pooler_output, split_sizes))
        coupler.reset_pools()
        for index in range(len(image_embeds) - 1):
            image_a = image_embeds[index].unsqueeze(0).to(torch.bfloat16)
            image_b = image_embeds[index + 1].unsqueeze(0).to(torch.bfloat16)
            refined_b = coupler(image_a, image_b)
            image_embeds[index + 1] = refined_b.squeeze(0).to(output.pooler_output.dtype)
        output.pooler_output = torch.cat(image_embeds, dim=0)
        return output
    for name, module in model.named_modules():
        if name.endswith("model.visual") or name == "visual":
            module.register_forward_hook(coupler_hook, with_kwargs=True)
            return name
    raise RuntimeError("visual tower not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["base", "lora", "icva"])
    ap.add_argument("--backbone", default="gemma", choices=["gemma", "qwen"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoProcessor
    if a.backbone == "gemma":
        from transformers import AutoModelForMultimodalLM
        mp = "google/gemma-4-E4B-it"
        model = AutoModelForMultimodalLM.from_pretrained(
            mp, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
        lora = os.path.join(CKPT_ROOT, "gemma_lora_only_step40000")
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration
        mp = "Qwen/Qwen2.5-VL-3B-Instruct"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            mp, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        lora = os.path.join(CKPT_ROOT, "qwen_lora_only_step10000")
    proc = AutoProcessor.from_pretrained(mp, trust_remote_code=True)
    CK = os.path.join(CKPT_ROOT, "qwen_icva_step10000")
    if a.model == "lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora).eval()
    elif a.model == "icva":
        assert a.backbone == "qwen", "icva arm wired for qwen only"
        from peft import PeftModel
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
        from model_gca_v3 import RecurrentCrossViewCoupler
        model = PeftModel.from_pretrained(model, CK).eval()
        hs = model.base_model.model.config.text_config.hidden_size
        coupler = RecurrentCrossViewCoupler(hs, num_iterations=3).to("cuda", torch.bfloat16)
        st = torch.load(f"{CK}/training_state.pt", map_location="cpu")
        coupler.load_state_dict(st["coupler_state"], strict=True)
        coupler.eval()
        hooked = _register_hook(model, coupler)
        print(f"  coupler hooked on {hooked}; alpha={float(coupler.output_scale):.6f}", flush=True)

    rows = load_rows(a.n, a.skip, a.stride)
    hits, preds, n = 0, collections.Counter(), 0
    for r in rows:
        imgs = [to_pil(x) for x in r["images"]]
        if a.backbone == "gemma": imgs = [im.resize((448, 448)) for im in imgs]
        text = proc.apply_chat_template(build_messages(r, len(imgs)), add_generation_prompt=True)
        inp = proc(text=text, images=imgs, return_tensors="pt").to("cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            g = model.generate(**inp, max_new_tokens=96, do_sample=False)
        pred = proc.decode(g[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        gold = str(r["correct_answer"]).strip()
        if gold.upper() in pred.upper(): hits += 1
        preds[pred[:40]] += 1; n += 1
        if n % 50 == 0: print(f"  {n}/{len(rows)}  acc={100*hits/n:.2f}", flush=True)
    rep = {"backbone": a.backbone, "arm": a.model, "skip": a.skip, "stride": a.stride, "n": n, "accuracy": round(100 * hits / n, 2),
           "top_predictions": preds.most_common(10)}
    json.dump(rep, open(a.out, "w"), indent=2)
    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()
