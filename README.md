# Jetson Orin StreamDiffusion

Real-time camera style transfer on the **NVIDIA Jetson Orin Nano Super**, using StreamDiffusion (SD-Turbo) accelerated with TensorRT.

CSI camera → SD-Turbo (StreamDiffusion) → TensorRT-accelerated UNet & TAESD decoder → output

Built on [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) and [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo). Uses [TAESD](https://github.com/madebyollin/taesd) as a lightweight VAE.

## Software
- JetPack 7.2.1 (L4T 39.2), CUDA 13.2, TensorRT 10.16, cuDNN 9.20
- Python 3.12, PyTorch 2.12 (community-built wheel for CUDA 13.2 / sm_87)
- 7.4GB unified memory + 16GB swap

> Previously developed on JetPack 6.2.1 (CUDA 12.6 / TensorRT 10.7); migrated to 7.2.1.

## Quick Start

### Prerequisites (on the device)
- JetPack 7.2.1
- A 16GB swapfile (needed when building TensorRT engines):
```bash
  sudo fallocate -l 16G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```
- [uv](https://github.com/astral-sh/uv) installed:
```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Install dependencies (one command)
```bash
git clone https://github.com/Spacemen2/jetson-orin-streamdiffusion.git
cd jetson-orin-streamdiffusion

# Create a venv that reuses the system TensorRT & OpenCV
uv venv .venv --python 3.12 --system-site-packages
source .venv/bin/activate

# Install the Jetson wheels + diffusion stack
uv pip install -r requirements.txt

# Install StreamDiffusion
uv pip install --no-deps \
  "streamdiffusion @ git+https://github.com/cumulo-autumn/StreamDiffusion.git@b623251dc055e1fd858d53509aa43e09dfc5cdc0"
```


### 2. Build the TensorRT engines (on-device, one-time)
```bash
source .venv/bin/activate

# Export ONNX from the SD-Turbo UNet and the TAESD decoder
python export_unet_onnx_jetson.py
python export_taesd_decoder_onnx_jetson.py

# Build the TensorRT engines (fp16)
python build_unet_engine.py  --onnx onnx_out/unet.opt.onnx      --engine engines/unet/unet_512_b1.engine  --batch 1 --height 512 --width 512
python build_taesd_engine.py --onnx onnx_out/taesd_decoder.onnx --engine engines/vae/taesd_512_b1.engine  --batch 1 --height 512 --width 512
```

### 3. Run
```bash
python diffusion_webcam.py
```
Press **q** to quit.

> First run downloads the SD-Turbo / TAESD weights from Hugging Face. After
> they are cached, set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for a
> faster, network-free startup.


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