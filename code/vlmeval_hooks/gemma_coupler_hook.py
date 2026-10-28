"""Gemma-4 coupler application site, extracted from our VLMEvalKit wrapper
(vlmeval/vlm/gemma.py).

The hook is registered on the visual projector, `embed_vision.embedding_projection`
-- the same module the training script hooks (train_v3.py), so the coupler is
applied at an identical site during training and evaluation.

Its input is the image-token stream only, shaped [num_images * 256, hidden].
Adjacent 256-token image chunks are refined sequentially, each against its
already-refined predecessor. Inputs carrying fewer than two images leave the
stream untouched -- the single-image bypass of Section 3.2.4. `coupler_enabled`
provides the matched inference-time bypass of Section 4.6 while keeping the base
model, LoRA weights, processor and decoding path identical.
"""

import logging
import torch


def make_coupler_hook(owner):
    """`owner` carries .coupler, .coupler_enabled and the two counters."""

    def coupler_hook(module, inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        image_token_len = 256

        # Only act on a 2-D image-token stream whose length is a whole
        # number of images; anything else passes through unchanged.
        if tensor.ndim != 2 or tensor.shape[0] % image_token_len:
            return output

        num_images = tensor.shape[0] // image_token_len
        # Single-image bypass, and the matched-control bypass.
        if not owner.coupler_enabled or num_images < 2:
            return output

        chunks = list(torch.split(tensor, image_token_len, dim=0))
        owner.coupler.reset_pools()
        owner.coupler_calls += 1
        for index in range(num_images - 1):
            image_a = chunks[index].unsqueeze(0).to(torch.bfloat16)
            image_b = chunks[index + 1].unsqueeze(0).to(torch.bfloat16)
            # Written back in place: pair (i+1, i+2) reads the refined view i+1.
            chunks[index + 1] = owner.coupler(
                image_a, image_b
            ).squeeze(0).to(tensor.dtype)
            owner.coupler_pairs += 1

        coupled = torch.cat(chunks, dim=0)
        return (coupled,) + output[1:] if isinstance(output, tuple) else coupled

    return coupler_hook


def register_gemma_coupler_hook(owner):
    for name, module in owner.model.named_modules():
        if 'embed_vision.embedding_projection' in name:
            module.register_forward_hook(make_coupler_hook(owner))
            logging.info('Registered Gemma visual hook on %s (enabled=%s)',
                         name, owner.coupler_enabled)
            return name
    raise RuntimeError('Could not find Gemma embed_vision.embedding_projection')
