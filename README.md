# Jetson Orin StreamDiffusion
 
Real-time webcam style transfer on the **NVIDIA Jetson Orin Nano Super**, using StreamDiffusion (SD-Turbo) accelerated with TensorRT.

USB webcam → SD-Turbo (StreamDiffusion) → TensorRT-accelerated UNet & VAE→ styled output

Built on [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) and [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo). Uses [TAESD](https://github.com/madebyollin/taesd) as a lightweight VAE.

## Software
- JetPack 6.2.1 (L4T 36.4.4), CUDA 12.6, TensorRT 10.7, cuDNN 9.3
- 7.4GB unified memory + 16GB swap (required for builds)

## Performance
 
| Stage | FPS | Notes |
|---|---|---|
| Naive PyTorch baseline | 0.76 | No TensorRT, full VAE |
| UNet → TensorRT (384×384, 2-step) | 1.60 | VAE still full-size |
| VAE → TAESD | 4.45 | Stable baseline |
| 512×512, single-step denoising | 4.16 | Higher resolution, fewer denoising steps |
| TAESD decoder → TensorRT | **4.9** | Current best |