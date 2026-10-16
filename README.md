# Jetson Orin StreamDiffusion

Real-time camera style transfer on the **NVIDIA Jetson Orin Nano Super**, using StreamDiffusion (SD-Turbo) accelerated with TensorRT.

CSI camera (IMX219) → SD-Turbo (StreamDiffusion) → TensorRT-accelerated UNet & TAESD decoder → styled output

Built on [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) and [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo). Uses [TAESD](https://github.com/madebyollin/taesd) as a lightweight VAE.

## Software
- JetPack 7.2.1 (L4T 39.2), CUDA 13.2, TensorRT 10.16, cuDNN 9.20
- Python 3.12, PyTorch 2.12 (community-built wheel for CUDA 13.2 / sm_87)
- 7.4GB unified memory + 16GB swap

> Previously developed on JetPack 6.2.1 (CUDA 12.6 / TensorRT 10.7); migrated to 7.2.1.

## Performance

Optimization progression (measured on JetPack 6.2.1):

| Stage | Resolution | Steps | FPS | Notes |
|---|---|---|---|---|
| Naive PyTorch baseline | 384×384 | 2 | 0.76 | No TensorRT, full VAE |
| UNet → TensorRT | 384×384 | 2 | 1.60 | VAE still full-size |
| VAE → TAESD | 384×384 | 2 | 4.45 | Lightweight VAE |
| Single-step, higher res | 512×512 | 1 | 4.16 | — |
| TAESD decoder → TensorRT | 512×512 | 1 | **4.9** | Best on 6.2.1 |

On **JetPack 7.2.1** (CUDA 13.2 / TensorRT 10.16), the same pipeline reaches **~5.1 FPS**.