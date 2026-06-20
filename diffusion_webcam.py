import time
import os
import torch
import cv2
import numpy as np
from PIL import Image

os.environ["HF_HOME"] = "/home/nobunali/hf_cache"

import tensorrt as trt
from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderTiny
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

ENGINE_PATH = "/home/nobunali/jetson-realtime-diffusion/unet_512_b1.engine"   # pre-built batch=1 TRT engine
TAESD_ENGINE_PATH = "/home/nobunali/jetson-realtime-diffusion/taesd_decoder.engine"

# ─────────────────────────────────────────────────────────────
# Wraps a TRT engine to replace stream.unet inside StreamDiffusion.
# StreamDiffusion calls self.unet(sample, timestep,
# encoder_hidden_states=..., return_dict=False)[0] during denoising,
# so this class just needs to accept those inputs, run the engine,
# and return (pred,).
# ─────────────────────────────────────────────────────────────
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
}


class TRTUNet(torch.nn.Module):
    def __init__(self, engine_path, ref_unet, device="cuda"):
        super().__init__()
        # StreamDiffusion may read these unet attributes; copy them from the original unet
        self.config = ref_unet.config
        self.dtype = ref_unet.dtype
        self.device = torch.device(device)

        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_names, self.output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        self.acc_ms = 0.0   # profiling: accumulated UNet GPU time per frame

        self.output_buffers = {}
        for name in self.output_names:
            shape = self.engine.get_tensor_shape(name)
            shape = tuple(d if d > 0 else 1 for d in shape)
            self.output_buffers[name] = torch.empty(
                shape, dtype=self._torch_dtype(name), device=self.device
            )

    def _torch_dtype(self, name):
        return TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]

    @torch.no_grad()
    def forward(self, sample, timestep, encoder_hidden_states=None,
                return_dict=False, **kwargs):
        batch = sample.shape[0]

        # normalize timestep to a [batch] tensor
        ts = timestep
        if not torch.is_tensor(ts):
            ts = torch.tensor(ts, device=self.device)
        ts = ts.reshape(-1)
        if ts.shape[0] != batch:
            ts = ts.expand(batch)

        feeds = {
            "sample": sample,
            "timestep": ts,
            "encoder_hidden_states": encoder_hidden_states,
        }

        # bind inputs: cast to engine-required dtype, set shape and device memory address
        keepalive = []
        for name in self.input_names:
            t = feeds[name].to(device=self.device,
                               dtype=self._torch_dtype(name)).contiguous()
            keepalive.append(t)
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())

        # allocate outputs
        outputs = {}
        for name in self.output_names:
            o = self.output_buffers[name]
            outputs[name] = o
            self.context.set_tensor_address(name, o.data_ptr())

        # run on current torch CUDA stream + time it
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        ev1.record()
        ev1.synchronize()
        self.acc_ms += ev0.elapsed_time(ev1)

        return (outputs["latent"],)

# ─────────────────────────────────────────────────────────────
# 把 TAESD decoder 的 TRT 引擎包一层，替换 stream.vae.decode。
# StreamDiffusion 每帧调用 self.vae.decode(latent, return_dict=False)[0]，
# 所以这个类接住 latent，跑引擎，按原接口返回 (image,)。
# ─────────────────────────────────────────────────────────────
class TRTTAESDDecoder(torch.nn.Module):
    def __init__(self, engine_path, device="cuda"):
        super().__init__()
        self.device = torch.device(device)

        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_names, self.output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def _torch_dtype(self, name):
        return TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]

    @torch.no_grad()
    def decode(self, latent, return_dict=True, **kwargs):
        t = latent.to(device=self.device,
                      dtype=self._torch_dtype("latent")).contiguous()
        self.context.set_input_shape("latent", tuple(t.shape))
        self.context.set_tensor_address("latent", t.data_ptr())

        out_shape = tuple(self.context.get_tensor_shape("image"))
        out = torch.empty(out_shape, dtype=self._torch_dtype("image"),
                          device=self.device)
        self.context.set_tensor_address("image", out.data_ptr())

        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)

        if return_dict:
            from diffusers.models.autoencoders.vae import DecoderOutput
            return DecoderOutput(sample=out)
        return (out,)

    def forward(self, latent):
        return self.decode(latent, return_dict=False)[0]

# ── Set up StreamDiffusion normally (VAE / text encoder remain in PyTorch) ──
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None,
).to("cuda")

stream = StreamDiffusion(
    pipe,
    t_index_list=[30],   # length 1 → UNet batch=1, matches engine; values are tunable
    torch_dtype=torch.float16,
    cfg_type="none",
    width=512,
    height=512,
)

# Switch to TAESD (lightweight VAE) — the full VAE is the current biggest bottleneck (~458ms/frame)
stream.vae = AutoencoderTiny.from_pretrained(
    "madebyollin/taesd"
).to(device="cuda", dtype=torch.float16)
del stream.vae.decoder
stream.vae.decode = TRTTAESDDecoder(TAESD_ENGINE_PATH, device="cuda").decode
print("[info] VAE changed to TAESD")

stream.prepare(
    prompt="Showa Japan 1980s Tokyo street, 35mm film photography, Kodachrome, shallow depth of field, natural lighting, photorealistic, shot on Canon AE-1",
    negative_prompt="anime, cartoon, illustration, painting, drawing, artistic, unrealistic, blur, low quality",
    num_inference_steps=50,
)

# ── Replace UNet with the TRT engine (safest to swap after prepare()) ──
stream.unet = TRTUNet(ENGINE_PATH, stream.unet, device="cuda")
print("[info] UNet changed to TRT engine")

cap = cv2.VideoCapture(0)
#cv2.namedWindow("StreamDiffusion", cv2.WINDOW_NORMAL)
#cv2.setWindowProperty("StreamDiffusion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.namedWindow("StreamDiffusion", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow("StreamDiffusion", 800, 480)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
LATENT_ALPHA = 0.2
prev_latent = None

prof = {k: 0.0 for k in ["preprocess", "unet", "vae+sched", "postprocess", "total"]}
nframes = 0
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    torch.cuda.synchronize(); t0 = time.perf_counter()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb).resize((512, 512))
    input_tensor = torch.from_numpy(np.array(frame_pil)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    input_tensor = input_tensor.to("cuda").half()
    torch.cuda.synchronize(); t1 = time.perf_counter()

    stream.unet.acc_ms = 0.0
    output = stream(input_tensor)
    #print(type(output), output.shape if hasattr(output, 'shape') else type(output))
    if prev_latent is not None and output is not None:
        output = LATENT_ALPHA * prev_latent + (1 - LATENT_ALPHA) * output
    prev_latent = output.detach().clone() if output is not None else None
    torch.cuda.synchronize(); t2 = time.perf_counter()

    if output is None:
        continue
    img = postprocess_image(output, output_type="pil")[0]
    torch.cuda.synchronize(); t3 = time.perf_counter()

    prof["preprocess"]  += (t1 - t0) * 1000
    prof["unet"]        += stream.unet.acc_ms
    prof["vae+sched"]   += (t2 - t1) * 1000 - stream.unet.acc_ms
    prof["postprocess"] += (t3 - t2) * 1000
    prof["total"]       += (t3 - t0) * 1000
    nframes += 1

    if nframes >= 30:
        a = {k: v / nframes for k, v in prof.items()}
        fps = 1000.0 / a["total"]
        print(f"[prof] total {a['total']:.0f}ms | unet {a['unet']:.0f} | "
              f"vae+sched {a['vae+sched']:.0f} | pre {a['preprocess']:.0f} | "
              f"post {a['postprocess']:.0f}  ->  {fps:.2f} FPS")
        prof = {k: 0.0 for k in prof}
        nframes = 0

    output_np = np.array(img)
    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    canvas = np.zeros((480, 800, 3), dtype=np.uint8)
    scale = 480 / 512
    new_size = int(512 * scale)  # = 480
    output_bgr = cv2.resize(output_bgr, (new_size, new_size), interpolation=cv2.INTER_CUBIC)
    x = (800 - new_size) // 2
    y = (480 - new_size) // 2
    canvas[y:y+new_size, x:x+new_size] = output_bgr
    output_bgr = canvas
    cv2.putText(output_bgr, f"{fps:4.1f} FPS", (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("StreamDiffusion", output_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
