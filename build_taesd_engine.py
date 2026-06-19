import argparse
import tensorrt as trt
 
 
def build_engine(onnx_path, engine_path, batch, height, width,
                 workspace_mb=256, opt_level=3, fp16=True):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
 
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("[ONNX]", parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")
 
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 workspace_mb * (1 << 20))
    config.builder_optimization_level = opt_level
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
 
    lat_h, lat_w = height // 8, width // 8
    profile = builder.create_optimization_profile()
    # latent: [batch, 4, H/8, W/8] -> image: [batch, 3, H, W]
    profile.set_shape("latent",
                      (batch, 4, lat_h, lat_w),
                      (batch, 4, lat_h, lat_w),
                      (batch, 4, lat_h, lat_w))
    config.add_optimization_profile(profile)
 
    print(f"[build] workspace={workspace_mb}MB opt_level={opt_level} "
          f"fp16={fp16} batch={batch} latent={lat_h}x{lat_w} -> image={height}x{width}")
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
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--workspace-mb", type=int, default=256)
    p.add_argument("--opt-level", type=int, default=3)
    p.add_argument("--no-fp16", action="store_true")
    a = p.parse_args()
    build_engine(a.onnx, a.engine, a.batch, a.height, a.width,
                 a.workspace_mb, a.opt_level, fp16=not a.no_fp16)