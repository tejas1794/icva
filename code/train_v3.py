#!/usr/bin/env python3
"""
train_v3.py - I-CVA Coupler Training with sliding window pairs and spatial auxiliary loss.
Compatible with Qwen2.5-VL and Gemma-4.
"""

import argparse
import io
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_dataset, load_from_disk
from PIL import Image
from torch.amp import autocast

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


def get_model_type(model_id):
    mid = model_id.lower()
    if "qwen" in mid:
        return "qwen"
    elif "gemma" in mid:
        return "gemma"
    else:
        raise ValueError(f"Unknown model type for {model_id}")


def load_model_and_processor(model_id, model_type, hf_token, use_gradient_checkpointing=False):
    """Load model in 4-bit quantized mode."""
    from transformers import BitsAndBytesConfig, AutoProcessor

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    if model_type == "qwen":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
            token=hf_token,
        )
        processor = AutoProcessor.from_pretrained(
            model_id, 
            token=hf_token,
            min_pixels=256*28*28,
            max_pixels=512*28*28
        )
    elif model_type == "gemma":
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,
        )
        processor = AutoProcessor.from_pretrained(model_id, token=hf_token)

    return model, processor


def setup_lora(model, model_type, use_gradient_checkpointing=False):
    """Configure LoRA adapters."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)

    if model_type == "qwen":
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    elif model_type == "gemma":
        target_modules = ["q_proj.linear", "v_proj.linear", "k_proj.linear", "o_proj.linear"]

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"LoRA: {trainable:,} trainable / {total:,} total params ({100*trainable/total:.2f}%)")
    return model


def setup_coupler(model, model_type, hidden_size):
    """Set up the Recurrent Cross-View Coupler and register hooks."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model_gca_v3 import RecurrentCrossViewCoupler

    coupler = RecurrentCrossViewCoupler(hidden_size=hidden_size, num_iterations=3).to(
        dtype=torch.bfloat16, device="cuda"
    )

    if model_type == "qwen":
        def coupler_hook(module, args_in, kwargs_in, output):
            grid_thw = kwargs_in.get("grid_thw", None)
            if grid_thw is None and len(args_in) > 1:
                grid_thw = args_in[1]
            if grid_thw is not None:
                spatial_merge_size = module.spatial_merge_size
                split_sizes = (grid_thw.prod(-1) // spatial_merge_size**2).tolist()
                N = len(split_sizes)
                if N >= 2:
                    image_embeds = list(torch.split(output.pooler_output, split_sizes))
                    # Sliding window coupling: (0, 1), (1, 2), etc.
                    for i in range(N - 1):
                        img_a = image_embeds[i].unsqueeze(0).to(torch.bfloat16)
                        img_b = image_embeds[i+1].unsqueeze(0).to(torch.bfloat16)
                        refined_b = coupler(img_a, img_b)
                        image_embeds[i+1] = refined_b.squeeze(0).to(output.pooler_output.dtype)
                    output.pooler_output = torch.cat(image_embeds, dim=0)
            return output

        patched = False
        for name, module in model.named_modules():
            if name.endswith("model.visual") or name == "visual":
                module.register_forward_hook(coupler_hook, with_kwargs=True)
                log.info(f"✅ Hooked Qwen visual tower: {name}")
                patched = True
                break
        if not patched:
            raise ValueError("Could not find visual module in Qwen model!")

    elif model_type == "gemma":
        def coupler_hook(module, input, output):
            t = output[0] if isinstance(output, tuple) else output
            total_patches, h = t.shape
            image_token_len = 256 # Fixed tokens per image in Gemma-4-E4B-it
            
            if total_patches % image_token_len == 0:
                N = total_patches // image_token_len
                if N >= 2:
                    chunks = list(torch.split(t, image_token_len, dim=0))
                    # Sliding window coupling
                    for i in range(N - 1):
                        img_a = chunks[i].unsqueeze(0).to(torch.bfloat16)
                        img_b = chunks[i+1].unsqueeze(0).to(torch.bfloat16)
                        refined_b = coupler(img_a, img_b)
                        chunks[i+1] = refined_b.squeeze(0).to(t.dtype)
                    t_coupled = torch.cat(chunks, dim=0)
                    return (t_coupled,) + output[1:] if isinstance(output, tuple) else t_coupled
            return output

        patched = False
        for name, module in model.named_modules():
            if "embed_vision.embedding_projection" in name:
                module.register_forward_hook(coupler_hook)
                log.info(f"✅ Hooked Gemma visual projector: {name}")
                patched = True
                break
        if not patched:
            raise ValueError("Could not find embedding projection module in Gemma model!")

    return coupler


def compute_spatial_loss(pool_a_tensor, pool_b_tensor, temperature=0.1):
    """Symmetric InfoNCE Loss for spatial alignment."""
    # Normalize features
    a_norm = F.normalize(pool_a_tensor, p=2, dim=-1)
    b_norm = F.normalize(pool_b_tensor, p=2, dim=-1)
    
    num_pairs = a_norm.size(0)
    if num_pairs > 1:
        # InfoNCE loss
        logits = torch.matmul(a_norm, b_norm.t()) / temperature  # [num_pairs, num_pairs]
        labels = torch.arange(num_pairs, device=a_norm.device)
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_a + loss_b)
    else:
        # Fallback to simple cosine distance if only 1 pair is present
        return 1.0 - torch.sum(a_norm * b_norm)


def build_messages(sample, num_images):
    """Build chat messages from a dataset sample."""
    if "options" in sample or "choices" in sample or "answers" in sample:
        system = "You are a spatial reasoning engine. Answer the multiple-choice question by reasoning step-by-step, then state the exact correct option."
        options = sample.get("options", sample.get("choices", sample.get("answers", [])))
        q_text = f"{sample['question']}\nOptions:\n" + "\n".join(options)
    else:
        system = "Analyze the multi-image sequence. Provide a detailed geometric evaluation of camera changes, depth shifts, and object positions."
        q_text = sample["question"]

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": system},
            *[{"type": "image"} for _ in range(num_images)],
            {"type": "text", "text": q_text},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": sample.get("correct_answer", sample.get("answer", ""))},
        ]},
    ]
    return messages


def mask_prompt_labels(input_ids, processor, model_type):
    """Mask prompt tokens in labels so loss is only on assistant response."""
    labels = input_ids.clone()
    token_ids = input_ids[0].tolist()

    if model_type == "qwen":
        assistant_text = "assistant\n"
        try:
            encoded_assistant = processor.tokenizer.encode(assistant_text, add_special_tokens=False)
            for i in range(len(token_ids) - len(encoded_assistant), -1, -1):
                if token_ids[i:i + len(encoded_assistant)] == encoded_assistant:
                    labels[0, :i + len(encoded_assistant)] = -100
                    break
            else:
                cutoff = int(len(token_ids) * 0.7)
                labels[0, :cutoff] = -100
        except Exception:
            cutoff = int(len(token_ids) * 0.7)
            labels[0, :cutoff] = -100
    elif model_type == "gemma":
        try:
            marker = processor.tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)
            for i in range(len(token_ids) - len(marker), -1, -1):
                if token_ids[i:i + len(marker)] == marker:
                    labels[0, :i + len(marker)] = -100
                    break
            else:
                cutoff = int(len(token_ids) * 0.7)
                labels[0, :cutoff] = -100
        except Exception:
            cutoff = int(len(token_ids) * 0.7)
            labels[0, :cutoff] = -100

    return labels


def prepare_validation_set(dataset_name, hf_token, num_samples=500, seed=42):
    """Collect a fixed validation set of multi-image samples."""
    log.info(f"Building validation set ({num_samples} multi-image samples)...")
    rng = random.Random(seed)

    ds = load_dataset(dataset_name, split="train", streaming=True, token=hf_token)
    val_samples = []
    candidates = []

    for i, sample in enumerate(ds):
        images = [img for img in sample.get("images", []) if img]
        if len(images) >= 2:
            candidates.append(sample)
        if len(candidates) >= num_samples * 3:
            break

    if len(candidates) > num_samples:
        val_samples = rng.sample(candidates, num_samples)
    else:
        val_samples = candidates

    log.info(f"Validation set: {len(val_samples)} samples collected")
    return val_samples


def run_validation(model, coupler, processor, val_samples, model_type, num_eval=50, seed=42):
    """Run validation on a subset of held-out samples. Returns (avg_loss, mcq_accuracy)."""
    model.eval()
    coupler.eval()
    rng = random.Random(seed)

    subset = rng.sample(val_samples, min(num_eval, len(val_samples)))
    total_loss = 0.0
    correct = 0
    total_mcq = 0
    valid_count = 0

    with torch.no_grad():
        for sample in subset:
            try:
                images = [
                    Image.open(io.BytesIO(img)) if isinstance(img, bytes) else img
                    for img in sample.get("images", []) if img
                ]
                if len(images) < 2:
                    continue

                # Cap images to 4 for memory safety
                images = images[:4]

                messages = build_messages(sample, len(images))
                prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
                inputs = processor(text=prompt, images=images, return_tensors="pt").to("cuda")
                inputs["labels"] = mask_prompt_labels(inputs.input_ids, processor, model_type)

                # Reset coupler pools
                coupler.reset_pools()

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(**inputs)
                
                lm_loss = outputs.loss
                
                # Check spatial auxiliary loss
                spatial_loss = torch.tensor(0.0, device="cuda")
                if len(coupler.collected_pools) > 0:
                    p_a = torch.cat([p[0] for p in coupler.collected_pools], dim=0)
                    p_b = torch.cat([p[1] for p in coupler.collected_pools], dim=0)
                    spatial_loss = compute_spatial_loss(p_a, p_b)

                # Combine losses in validation as well
                total_loss += (lm_loss + 0.1 * spatial_loss).item()
                valid_count += 1

                # MCQ accuracy check
                if "options" in sample or "choices" in sample:
                    answer = sample.get("correct_answer", sample.get("answer", ""))
                    gen_messages = [{"role": "user", "content": messages[0]["content"]}]
                    gen_prompt = processor.apply_chat_template(gen_messages, add_generation_prompt=True)
                    gen_inputs = processor(text=gen_prompt, images=images, return_tensors="pt").to("cuda")
                    
                    coupler.reset_pools()
                    gen_ids = model.generate(**gen_inputs, max_new_tokens=32, do_sample=False)
                    pred = processor.decode(gen_ids[0][gen_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                    if answer.strip().upper() in pred.upper()[:10]:
                        correct += 1
                    total_mcq += 1

            except Exception as e:
                log.warning(f"Validation sample error: {e}")
                continue

    model.train()
    coupler.train()

    avg_loss = total_loss / max(valid_count, 1)
    accuracy = correct / max(total_mcq, 1) if total_mcq > 0 else None
    return avg_loss, accuracy, valid_count


def save_checkpoint(save_path, model, coupler, optimizer, scheduler, step, val_history):
    """Save full checkpoint for resume."""
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    torch.save({
        "coupler_state": coupler.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "step": step,
        "val_history": val_history,
    }, os.path.join(save_path, "training_state.pt"))
    log.info(f"💾 Checkpoint saved: {save_path} (step {step})")


def load_checkpoint(save_dir, model, coupler, optimizer, scheduler):
    """Find and load latest checkpoint."""
    if not os.path.isdir(save_dir):
        return 0, []

    checkpoints = []
    for d in os.listdir(save_dir):
        if d.startswith("checkpoint_"):
            try:
                step = int(d.split("_")[1])
                state_path = os.path.join(save_dir, d, "training_state.pt")
                if os.path.exists(state_path):
                    checkpoints.append((step, os.path.join(save_dir, d)))
            except (ValueError, IndexError):
                continue

    if not checkpoints:
        return 0, []

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    latest_step, latest_path = checkpoints[0]

    log.info(f"📂 Resuming from checkpoint: {latest_path} (step {latest_step})")

    from peft import set_peft_model_state_dict
    adapter_path = latest_path
    
    safetensors_file = os.path.join(adapter_path, "adapter_model.safetensors")
    bin_file = os.path.join(adapter_path, "adapter_model.bin")
    
    if os.path.exists(safetensors_file):
        import safetensors.torch
        adapters_weights = safetensors.torch.load_file(safetensors_file, device="cuda")
        set_peft_model_state_dict(model, adapters_weights)
        log.info("✅ LoRA adapter weights loaded (safetensors)")
    elif os.path.exists(bin_file):
        adapters_weights = torch.load(bin_file, map_location="cuda", weights_only=True)
        set_peft_model_state_dict(model, adapters_weights)
        log.info("✅ LoRA adapter weights loaded (bin)")

    state = torch.load(os.path.join(latest_path, "training_state.pt"), map_location="cuda", weights_only=False)
    coupler.load_state_dict(state["coupler_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    if scheduler and state.get("scheduler_state"):
        scheduler.load_state_dict(state["scheduler_state"])

    return state["step"], state.get("val_history", [])


def train(args):
    model_type = get_model_type(args.model_id)
    hidden_size = {"qwen": 2048, "gemma": 2560}.get(model_type, 2048) # Gemma-4-E4B projector is 2560

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model_id.split("/")[-1]
    run_name = args.run_name or f"{model_short}_icva_v3_{ts}"
    run_save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_save_dir, exist_ok=True)

    wb_mode = os.environ.get("WANDB_MODE", "offline")
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "icva"),
        name=run_name,
        config=vars(args),
        mode=wb_mode,
    )

    log.info(f"Loading {args.model_id}...")
    model, processor = load_model_and_processor(args.model_id, model_type, args.hf_token, args.gradient_checkpointing)
    model = setup_lora(model, model_type, args.gradient_checkpointing)
    coupler = setup_coupler(model, model_type, hidden_size)

    # Optimizer
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    coupler_params = list(coupler.parameters())
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr},
        {"params": coupler_params, "lr": args.lr * 5},
    ], weight_decay=0.01)

    warmup_steps = int(args.steps * args.warmup_ratio)
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, args.steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step, val_history = load_checkpoint(run_save_dir, model, coupler, optimizer, scheduler)

    val_samples = prepare_validation_set(args.dataset, args.hf_token, num_samples=500)

    log.info(f"Loading dataset: {args.dataset}")
    if os.path.isdir(args.dataset):
        ds = load_from_disk(args.dataset)
        if isinstance(ds, dict):
            ds = ds["train"]
    else:
        ds = load_dataset(args.dataset, split="train", streaming=True, token=args.hf_token)

    # Skip to resume position
    it = iter(ds)
    if start_step > 0:
        log.info(f"⏩ Skipping to step {start_step}...")
        skipped = 0
        while skipped < start_step:
            sample = next(it, None)
            if sample is None:
                break
            images = [img for img in sample.get("images", []) if img]
            if len(images) >= 2:
                skipped += 1

    model.train()
    coupler.train()
    step = start_step
    accum_lm_loss = 0.0
    accum_spatial_loss = 0.0
    step_times = []
    samples_skipped = 0
    consecutive_errors = 0

    log.info(f"🚀 Training from step {start_step} to {args.steps}")
    log.info(f"   LR: {args.lr} (LoRA) / {args.lr * 5} (Coupler)")
    log.info(f"   Warmup: {warmup_steps} steps, Accumulation: {args.accumulation_steps}")

    while step < args.steps:
        step_start = time.time()

        try:
            sample = next(it)
        except StopIteration:
            log.info("Dataset exhausted, restarting...")
            if os.path.isdir(args.dataset):
                ds = load_from_disk(args.dataset)
                if isinstance(ds, dict):
                    ds = ds["train"]
            else:
                ds = load_dataset(args.dataset, split="train", streaming=True, token=args.hf_token)
            it = iter(ds)
            continue

        try:
            images = [
                Image.open(io.BytesIO(img)) if isinstance(img, bytes) else img
                for img in sample.get("images", []) if img
            ]
            if len(images) < 2:
                samples_skipped += 1
                continue

            # Cap images to 4 for memory safety
            images = images[:4]

            messages = build_messages(sample, len(images))
            prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
            inputs = processor(text=prompt, images=images, return_tensors="pt").to("cuda")

            inputs["labels"] = mask_prompt_labels(inputs.input_ids, processor, model_type)

            # Reset coupler pools
            coupler.reset_pools()

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
                lm_loss = outputs.loss
                
                # Retrieve visual representations pooled by coupler
                spatial_loss = torch.tensor(0.0, device="cuda")
                if len(coupler.collected_pools) > 0:
                    p_a = torch.cat([p[0] for p in coupler.collected_pools], dim=0)
                    p_b = torch.cat([p[1] for p in coupler.collected_pools], dim=0)
                    spatial_loss = compute_spatial_loss(p_a, p_b)

                # Total combined loss
                loss = (lm_loss + 0.1 * spatial_loss) / args.accumulation_steps

            loss.backward()
            accum_lm_loss += lm_loss.item() / args.accumulation_steps
            accum_spatial_loss += spatial_loss.item() / args.accumulation_steps

            if (step + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(coupler.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step_time = time.time() - step_start
            step_times.append(step_time)

            if (step + 1) % 10 == 0:
                avg_time = sum(step_times[-100:]) / len(step_times[-100:])
                remaining = (args.steps - step - 1) * avg_time
                hrs, rem = divmod(remaining, 3600)
                mins = rem // 60

                full_lm_loss = accum_lm_loss * args.accumulation_steps
                full_spatial_loss = accum_spatial_loss * args.accumulation_steps
                total_loss = full_lm_loss + 0.1 * full_spatial_loss
                
                log.info(
                    f"Step {step+1}/{args.steps} | LM Loss: {full_lm_loss:.4f} | Spatial Loss: {full_spatial_loss:.4f} | "
                    f"Total: {total_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"{avg_time:.2f}s/step | Skipped: {samples_skipped}"
                )
                
                wandb.log({
                    "step": step + 1,
                    "train_lm_loss": full_lm_loss,
                    "train_spatial_loss": full_spatial_loss,
                    "train_total_loss": total_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "step_time": avg_time,
                    "samples_skipped": samples_skipped,
                })
                accum_lm_loss = 0.0
                accum_spatial_loss = 0.0

            # Validation
            if (step + 1) % args.val_interval == 0:
                log.info(f"📊 Running validation at step {step+1}...")
                val_loss, val_acc, val_n = run_validation(
                    model, coupler, processor, val_samples, model_type,
                    num_eval=args.val_samples,
                )
                val_entry = {"step": step + 1, "val_loss": val_loss, "val_acc": val_acc}
                val_history.append(val_entry)

                log_dict = {"step": step + 1, "val_loss": val_loss, "val_n": val_n}
                acc_str = "N/A"
                if val_acc is not None:
                    log_dict["val_mcq_accuracy"] = val_acc
                    acc_str = f"{val_acc*100:.1f}%"
                wandb.log(log_dict)

                log.info(f"   Val Loss: {val_loss:.4f} | MCQ Acc: {acc_str} | Samples: {val_n}")

            # Checkpoint
            if (step + 1) % args.checkpoint_interval == 0:
                ckpt_path = os.path.join(run_save_dir, f"checkpoint_{step+1}")
                save_checkpoint(ckpt_path, model, coupler, optimizer, scheduler, step + 1, val_history)

            consecutive_errors = 0
            step += 1

        except torch.cuda.OutOfMemoryError:
            log.warning(f"OOM at step {step}, clearing cache and skipping...")
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            consecutive_errors += 1
            if consecutive_errors > 10:
                log.error("Too many consecutive errors/OOMs, aborting.")
                raise
            continue
        except Exception as e:
            consecutive_errors += 1
            log.warning(f"Error at step {step}: {e}")
            if consecutive_errors > 10:
                log.error("Too many consecutive errors, aborting.")
                raise e
            continue

    final_path = os.path.join(run_save_dir, f"checkpoint_{step}")
    save_checkpoint(final_path, model, coupler, optimizer, scheduler, step, val_history)
    wandb.finish()
    log.info(f"✅ Training complete. Checkpoints in: {run_save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="I-CVA Training v3")
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="google/gemma-4-E4B-it")
    parser.add_argument("--dataset", type=str, default="array/SAT-v2")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("--val_interval", type=int, default=2000)
    parser.add_argument("--val_samples", type=int, default=50)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--save_dir", type=str, default="./local_checkpoints")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--run_name", type=str, default="")
    train(parser.parse_args())
