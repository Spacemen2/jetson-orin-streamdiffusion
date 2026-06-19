import os
import time
import torch
from diffusers import AutoencoderTiny
 
# ── Config ──────────────────────────────────────────────
OUT_DIR = r"C:\sd_export\onnx"
LATENT_H = 64      # 512 // 8
LATENT_W = 64      # 512 // 8
SAMPLE_BATCH = 1   # decoder 每帧处理1张图，跟UNet的batch无关
# ─────────────────────────────────────────────────────────
 
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda")
dtype = torch.float16
 
taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd")
taesd = taesd.to(device=device, dtype=dtype).eval()
 
 
class TAESDDecoderWrapper(torch.nn.Module):
    """只暴露 decode，输出纯tensor（不要DecoderOutput包装类）"""
    def __init__(self, taesd):
        super().__init__()
        self.taesd = taesd
 
    def forward(self, latent):
        return self.taesd.decode(latent, return_dict=False)[0]
 
 
wrapper = TAESDDecoderWrapper(taesd).to(device).eval()
 
dummy = (
    torch.randn(SAMPLE_BATCH, 4, LATENT_H, LATENT_W, device=device, dtype=dtype),
)
 
onnx_path = os.path.join(OUT_DIR, "taesd_decoder.onnx")
print("[export] ->", onnx_path)
torch.cuda.empty_cache()
t0 = time.perf_counter()
with torch.no_grad():
    torch.onnx.export(
        wrapper, dummy, onnx_path,
        input_names=["latent"],
        output_names=["image"],
        dynamic_axes={
            "latent": {0: "batch", 2: "H", 3: "W"},
            "image":  {0: "batch", 2: "outH", 3: "outW"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
print(f"[ok] exported in {time.perf_counter()-t0:.1f}s:", onnx_path)
 
import onnx
onnx.checker.check_model(onnx.load(onnx_path))
print("[ok] ONNX validation passed")
print("\nNext: polygraphy inspect model", onnx_path)
print("Then build with build_unet_engine.py-style script (adjust input/output names)")
 