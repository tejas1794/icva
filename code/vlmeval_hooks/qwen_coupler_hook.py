"""Qwen2.5-VL coupler application site, extracted from our VLMEvalKit wrapper
(vlmeval/vlm/qwen2_vl/model.py, where Qwen2VLChat is defined).

The hook is registered on the visual tower and refines adjacent images
sequentially. Inputs with fewer than two images pass through unchanged, which is
the single-image bypass of Section 3.2.4.
"""

import logging
import torch


def _register_icva_qwen_hook(model, coupler):
    """Register the visual-tower coupling operation used by train_v3.py."""

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
            logging.info("Registered I-CVA Qwen visual hook on %s", name)
            return name
    raise RuntimeError("Could not find the Qwen visual tower for the I-CVA hook")


class ICVAQwen2_5_VL(Qwen2VLChat):
    """Coupler-only evaluator: base model plus coupler, no LoRA."""

    def __init__(self, model_path, coupler_path, icva_code_path="code", num_iterations=3, **kwargs):
        import sys
        import torch
        super().__init__(model_path=model_path, **kwargs)
        if icva_code_path not in sys.path:
            sys.path.insert(0, icva_code_path)
        from model_gca_v3 import RecurrentCrossViewCoupler
        self.coupler = RecurrentCrossViewCoupler(
            self.model.config.text_config.hidden_size, num_iterations=num_iterations
        ).to(device="cuda", dtype=torch.bfloat16)
        state = torch.load(coupler_path, map_location="cpu")
        self.coupler.load_state_dict(state["coupler_state"], strict=True)
        self.coupler.eval()
        self.icva_hook_module = _register_icva_qwen_hook(self.model, self.coupler)
        self.icva_provenance = {"coupler_path": coupler_path, "lora_path": None}
