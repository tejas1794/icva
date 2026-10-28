import os
import sys
import time
import torch
import numpy as np
from PIL import Image

# Our VLMEvalKit fork (model wrappers with the coupler attached)
sys.path.insert(0, os.environ["VLMEVALKIT_PATH"])

def run_benchmark(model_name):
    # Select appropriate class
    if model_name == "Gemma4":
        from vlmeval.vlm.gemma import Gemma4
        model = Gemma4()
    elif model_name == "ICVAGemma4":
        from vlmeval.vlm.gemma import ICVAGemma4
        model = ICVAGemma4()
    elif model_name == "Qwen2.5-VL":
        from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat
        model = Qwen2VLChat(model_path="Qwen/Qwen2.5-VL-3B-Instruct")
    elif model_name == "ICVAQwen2.5-VL":
        from vlmeval.vlm.qwen2_vl.model import ICVAQwen2_5_VL
        model = ICVAQwen2_5_VL(model_path="Qwen/Qwen2.5-VL-3B-Instruct", coupler_path=os.environ["COUPLER_CKPT"])
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Cap generation at 5 tokens so the measurement isolates vision encoding and prefill
    if hasattr(model, 'kwargs'):
        model.kwargs['max_new_tokens'] = 5
    if hasattr(model, 'generate_kwargs'):
        model.generate_kwargs['max_new_tokens'] = 5
        model.max_new_tokens = 5

    # Prepare inputs
    img_path = os.environ["LATENCY_IMAGE"]  # any RGB image; used twice to form a two-view input
    message = [
        {"type": "image", "value": img_path},
        {"type": "image", "value": img_path},
        {"type": "text", "value": "Are these two views of the same scene?"}
    ]

    # Warmup passes
    print(f"Warmup for {model_name}...")
    sys.stdout.flush()
    for _ in range(5):
        _ = model.generate_inner(message)

    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()

    # Latency benchmarking
    print(f"Benchmarking latency for {model_name}...")
    sys.stdout.flush()
    latencies = []
    for i in range(20):
        start = time.perf_counter()
        _ = model.generate_inner(message)
        end = time.perf_counter()
        latencies.append(end - start)

    mean_latency = np.mean(latencies) * 1000.0  # in ms
    std_latency = np.std(latencies) * 1000.0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)  # in GB

    print(f"\n=== RESULTS FOR {model_name} ===")
    print(f"Inference Latency (max_new_tokens=5): {mean_latency:.2f} ms ± {std_latency:.2f} ms")
    print(f"Peak VRAM during inference: {peak_vram:.2f} GB")
    sys.stdout.flush()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: LATENCY_IMAGE=img.jpg [COUPLER_CKPT=...] python measure_latency.py <model_name>")
        sys.exit(1)

    run_benchmark(sys.argv[1])
