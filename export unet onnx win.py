#!/usr/bin/env python3
import os
import torch
from diffusers import UNet2DConditionModel

# ── Config: set OUT_DIR to your local path ──────────────
MODEL_ID = "stabilityai/sd-turbo"
OUT_DIR = r"C:\sd_export\onnx"      # ← Windows path, use r"..." raw string

SAMPLE_BATCH = 2
LATENT_H = 48      # 384 // 8
LATENT_W = 48      # 384 // 8
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

# Dummy inputs: sample/enc in fp16 to match the model; timestep in fp32 (cast internally)
dummy = (
    torch.randn(SAMPLE_BATCH, 4, LATENT_H, LATENT_W, device=device, dtype=dtype),
    torch.randn(SAMPLE_BATCH, device=device, dtype=torch.float32),
    torch.randn(SAMPLE_BATCH, MAX_TOKENS, EMBED_DIM, device=device, dtype=dtype),
)

onnx_path = os.path.join(OUT_DIR, "unet.onnx")
print("[export] ->", onnx_path)
with torch.no_grad():
    torch.onnx.export(
        wrapper, dummy, onnx_path,
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["latent"],
        dynamic_axes={
            "sample":                {0: "2B", 2: "H", 3: "W"},
            "timestep":              {0: "2B"},
            "encoder_hidden_states": {0: "2B"},
            "latent":                {0: "2B", 2: "H", 3: "W"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
print("[ok] exported:", onnx_path)
print("\nNext: copy unet.onnx to engines/onnx/ on the Jetson,")