#!/usr/bin/env python3
"""
train_lora_only.py - LoRA-only Training (NO I-CVA Coupler).
Identical hyperparameters, data, and LoRA config as train_v3.py.
This is the critical ablation: same spatial training data, same LoRA,
but WITHOUT the geometric coupler module.
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
    """Configure LoRA adapters — identical to train_v3.py."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)

    if model_type == "qwen":
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    elif model_type == "gemma":
        target_modules = r".*language_model.*self_attn.*\.(q_proj|k_proj|v_proj|o_proj)$"

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


def build_messages(sample, num_images):
    """Build chat messages — identical to train_v3.py."""
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
    """Mask prompt tokens — identical to train_v3.py."""
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


def run_validation(model, processor, val_samples, model_type, num_eval=50):
    """Run validation — simplified (no coupler)."""
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for sample in val_samples[:num_eval]:
            try:
                images = [
                    Image.open(io.BytesIO(img)) if isinstance(img, bytes) else img
                    for img in sample.get("images", []) if img
                ]
                if len(images) < 2:
                    continue
                images = images[:4]

                messages = build_messages(sample, len(images))
                prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
                inputs = processor(text=prompt, images=images, return_tensors="pt").to("cuda")
                inputs["labels"] = mask_prompt_labels(inputs.input_ids, processor, model_type)

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(**inputs)
                    total_loss += outputs.loss.item()
                    count += 1
            except Exception:
                continue

    model.train()
    avg_loss = total_loss / max(count, 1)
    return avg_loss, None, count


def save_checkpoint(path, model, optimizer, scheduler, step, val_history):
    """Save checkpoint — no coupler state."""
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    torch.save({
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "step": step,
        "val_history": val_history,
    }, os.path.join(path, "training_state.pt"))
    log.info(f"💾 Saved checkpoint: {path} (step {step})")


def train(args):
    model_type = get_model_type(args.model_id)
    log.info(f"Model type: {model_type}")
    log.info(f"=== LoRA-ONLY ABLATION (NO COUPLER) ===")

    model, processor = load_model_and_processor(
        args.model_id, model_type, args.hf_token,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )
    model = setup_lora(model, model_type, use_gradient_checkpointing=args.gradient_checkpointing)

    # No coupler: this is the only architectural difference from train_v3.py

    run_name = args.run_name or f"lora_only_{model_type}_{datetime.now():%Y%m%d_%H%M%S}"
    run_save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_save_dir, exist_ok=True)

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "icva"),
        name=run_name,
        config=vars(args),
        mode=os.environ.get("WANDB_MODE", "offline"),
    )

    # Optimizer over LoRA parameters only
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    warmup_steps = int(args.steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(args.steps - warmup_steps, 1)
    )

    # Load validation data
    val_samples = []
    start_step = 0
    val_history = []

    # Check for resume
    existing_ckpts = []
    if os.path.exists(run_save_dir):
        for d in os.listdir(run_save_dir):
            if d.startswith("checkpoint_"):
                try:
                    step_num = int(d.split("_")[1])
                    existing_ckpts.append(step_num)
                except ValueError:
                    pass

    if existing_ckpts:
        latest = max(existing_ckpts)
        ckpt_path = os.path.join(run_save_dir, f"checkpoint_{latest}")
        log.info(f"⏩ Resuming from checkpoint: {ckpt_path}")

        from peft import PeftModel
        state = torch.load(os.path.join(ckpt_path, "training_state.pt"), map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_step = state["step"]
        val_history = state.get("val_history", [])
        log.info(f"   Resumed at step {start_step}")

    # Collect validation samples
    log.info(f"Loading dataset: {args.dataset}")
    if os.path.isdir(args.dataset):
        ds = load_from_disk(args.dataset)
        if isinstance(ds, dict):
            ds = ds["train"]
    else:
        ds = load_dataset(args.dataset, split="train", streaming=True, token=args.hf_token)

    log.info("Collecting validation samples...")
    val_iter = iter(ds)
    while len(val_samples) < args.val_samples * 2:
        s = next(val_iter, None)
        if s is None:
            break
        images = [img for img in s.get("images", []) if img]
        if len(images) >= 2:
            val_samples.append(s)

    # Reload for training
    if os.path.isdir(args.dataset):
        ds = load_from_disk(args.dataset)
        if isinstance(ds, dict):
            ds = ds["train"]
    else:
        ds = load_dataset(args.dataset, split="train", streaming=True, token=args.hf_token)

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
    step = start_step
    accum_lm_loss = 0.0
    step_times = []
    samples_skipped = 0
    consecutive_errors = 0

    log.info(f"🚀 LoRA-only training from step {start_step} to {args.steps}")
    log.info(f"   LR: {args.lr}")
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

            images = images[:4]

            messages = build_messages(sample, len(images))
            prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
            inputs = processor(text=prompt, images=images, return_tensors="pt").to("cuda")
            inputs["labels"] = mask_prompt_labels(inputs.input_ids, processor, model_type)


            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
                lm_loss = outputs.loss
                # Language-modeling loss only (no auxiliary term)
                loss = lm_loss / args.accumulation_steps

            loss.backward()
            accum_lm_loss += lm_loss.item() / args.accumulation_steps

            if (step + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
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

                log.info(
                    f"Step {step+1}/{args.steps} | LM Loss: {full_lm_loss:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"{avg_time:.2f}s/step | Skipped: {samples_skipped}"
                )

                wandb.log({
                    "step": step + 1,
                    "train_lm_loss": full_lm_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "step_time": avg_time,
                    "samples_skipped": samples_skipped,
                })
                accum_lm_loss = 0.0

            # Validation
            if (step + 1) % args.val_interval == 0:
                log.info(f"📊 Running validation at step {step+1}...")
                val_loss, val_acc, val_n = run_validation(
                    model, processor, val_samples, model_type,
                    num_eval=args.val_samples,
                )
                val_entry = {"step": step + 1, "val_loss": val_loss, "val_acc": val_acc}
                val_history.append(val_entry)

                wandb.log({"step": step + 1, "val_loss": val_loss, "val_n": val_n})
                log.info(f"   Val Loss: {val_loss:.4f} | Samples: {val_n}")

            # Checkpoint
            if (step + 1) % args.checkpoint_interval == 0:
                ckpt_path = os.path.join(run_save_dir, f"checkpoint_{step+1}")
                save_checkpoint(ckpt_path, model, optimizer, scheduler, step + 1, val_history)

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
    save_checkpoint(final_path, model, optimizer, scheduler, step, val_history)
    wandb.finish()
    log.info(f"✅ LoRA-only training complete. Checkpoints in: {run_save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA-Only Training (No I-CVA)")
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
