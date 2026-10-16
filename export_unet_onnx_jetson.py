#!/usr/bin/env python3
import os
import torch
import onnx
import subprocess
from diffusers import UNet2DConditionModel

# ── Config: set OUT_DIR to your local path ──────────────
MODEL_ID = "stabilityai/sd-turbo"
OUT_DIR = "onnx_out"      # ← Windows path, use r"..." raw string

SAMPLE_BATCH = 2
LATENT_H = 64      # 512 // 8
LATENT_W = 64      # 512 // 8
EMBED_DIM = 1024   # SD-Turbo
MAX_TOKENS = 77
# ──────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda")
dtype = torch.float16

unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
unet = unet.to(device=device, dtype=dtype).eval()


class UNetWrapper(torch.nn.Module):
    """torch.onnx requires tensor output; wrap to return only the sample."""
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet(sample, timestep, encoder_hidden_states,
                         return_dict=False)[0]


wrapper = UNetWrapper(unet).to(device).eval()

# Dummy inputs: scale fp16 tensors down to avoid NaN during trace; timestep uses realistic integer-like values
dummy = (
    torch.randn(SAMPLE_BATCH, 4, LATENT_H, LATENT_W, device=device, dtype=dtype) * 0.1,
    torch.tensor([999.0, 999.0], device=device, dtype=torch.float32),
    torch.randn(SAMPLE_BATCH, MAX_TOKENS, EMBED_DIM, device=device, dtype=dtype) * 0.1,
)

onnx_path = os.path.join(OUT_DIR, "unet.onnx")
opt_path = os.path.join(OUT_DIR, "unet.opt.onnx")

print("[export] ->", onnx_path)
with torch.no_grad():
    torch.onnx.export(
        wrapper, dummy, onnx_path,
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["latent"],
        dynamic_axes={
            "sample":                {0: "batch", 2: "H", 3: "W"},
            "timestep":              {0: "batch"},
            "encoder_hidden_states": {0: "batch"},
            "latent":                {0: "batch", 2: "H", 3: "W"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo = False,
    )
    
# —— Verify the ONNX model after export
model = onnx.load(onnx_path)
onnx.checker.check_model(model)

subprocess.run(
    ["polygraphy", "surgeon", "sanitize", onnx_path,
     "--fold-constants", "-o", opt_path],
    check=True,
)

opt_model = onnx.load(opt_path)
onnx.checker.check_model(opt_model)

print("[ok] exported:", onnx_path)
print("[ok] ONNX validation passed")
print("[ok] optimized & validated:", opt_path)
print("\nNext: copy unet.onnx to engines/onnx/ on the Jetson,")