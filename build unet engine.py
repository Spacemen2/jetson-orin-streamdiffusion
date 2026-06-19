#!/usr/bin/env python3
"""
Build a TensorRT engine from ONNX only.
No torch model is loaded in-process; peak memory = ONNX parsing + builder workspace.
Fully decoupled from ONNX export, eliminating OOM caused by overlapping memory peaks.

Usage example:
  python3 build_unet_engine.py \
      --onnx   /home/.../unet.opt.onnx \
      --engine /home/.../unet.engine \
      --batch 1 --height 512 --width 512 \
      --workspace-mb 512 --opt-level 2

Notes:
  * --batch must match the batch dimension StreamDiffusion actually feeds to UNet
    (typically = frame_buffer_size * len(t_index_list)); confirm with polygraphy first.
  * Input tensor names (sample/timestep/encoder_hidden_states) must match your exported
    ONNX -- run `polygraphy inspect model <onnx>` to check, then calibrate the TODO
    section below accordingly.
"""
import argparse
import tensorrt as trt


def build_engine(onnx_path, engine_path, batch, height, width,
                 embed_dim=1024, max_tokens=77,
                 workspace_mb=512, opt_level=2, fp16=True):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    # TRT 10: explicit batch is the default; create_network(0) is sufficient
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("[ONNX]", parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    # ── Key 1: cap workspace so TRT doesn't fill all unified memory by default ──
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 workspace_mb * (1 << 20))
    # ── Key 2: lower optimization level (0-5, default 3) to reduce build memory and time ──
    config.builder_optimization_level = opt_level
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    lat_h, lat_w = height // 8, width // 8
    profile = builder.create_optimization_profile()
    # ↓↓↓ TODO: calibrate input names / shapes against polygraphy output ↓↓↓
    profile.set_shape("sample",
                      (batch, 4, lat_h, lat_w),
                      (batch, 4, lat_h, lat_w),
                      (batch, 4, lat_h, lat_w))
    profile.set_shape("timestep", (batch,), (batch,), (batch,))
    profile.set_shape("encoder_hidden_states",
                      (batch, max_tokens, embed_dim),
                      (batch, max_tokens, embed_dim),
                      (batch, max_tokens, embed_dim))
    # ↑↑↑ TODO ↑↑↑
    config.add_optimization_profile(profile)

    print(f"[build] workspace={workspace_mb}MB opt_level={opt_level} "
          f"fp16={fp16} batch={batch} {height}x{width}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print("[ok]", engine_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--embed-dim", type=int, default=1024)   # SD-Turbo = 1024
    p.add_argument("--max-tokens", type=int, default=77)
    p.add_argument("--workspace-mb", type=int, default=512)
    p.add_argument("--opt-level", type=int, default=2)
    p.add_argument("--no-fp16", action="store_true")
    a = p.parse_args()
    build_engine(a.onnx, a.engine, a.batch, a.height, a.width,
                 a.embed_dim, a.max_tokens,
                 a.workspace_mb, a.opt_level, fp16=not a.no_fp16)