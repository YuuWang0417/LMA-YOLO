# LMA-YOLO: A Lightweight Manhattan Attention Network for Aortic Valve Detection in Cardiac CT Based on YOLOv12

This repository contains the training / evaluation code and model configuration for **LMA-YOLO**, a lightweight object detector for automated aortic valve localization in cardiac CT images. LMA-YOLO is built on top of **YOLOv12** and replaces the mixed Manhattan–Cosine attention design used in SMA-YOLO with a **pure Manhattan Self-Attention (A2C2fManhattan)** module, improving sensitivity to local grayscale intensity differences that are characteristic of low-contrast, boundary-blurred CT slices.

## Highlights

- Single-class object detector for **aortic valve** localization on cardiac CT.
- Custom `A2C2fManhattan` attention module integrated into the YOLOv12 detection head.
- Lightweight design: ~2.58M parameters, ~6.0 GFLOPs (nano scale).
- Evaluated against RCS-YOLO, PK-YOLO, YOLOv11, YOLOv12, and YOLO26 baselines.

## Repository Structure

```
.
├── aortic_valve_colab.yaml       # Dataset configuration
├── LICENSE
├── README.md
├── .gitignore
└── yolov12/
    ├── train.py                   # Training entry point
    ├── requirements.txt           # Python dependencies
    └── yolov12_LMA-YOLO.yaml      # LMA-YOLO model configuration (all scales)
```

## Hardware

Training/evaluation was performed on:
- NVIDIA GeForce RTX 4070 (12GB)
- NVIDIA GeForce RTX 3060 Ti (8GB)

## Environment Setup

Tested on Windows with an Anaconda environment (Python 3.10, CUDA 12.8).

```bash
conda create -n yolov12 python=3.10 -y
conda activate yolov12

pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

Verify the GPU is detected:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Dependencies

| Package | Version |
|---|---|
| ultralytics | 8.4.83 |
| ultralytics-thop | 2.0.20 |
| thop | 0.1.1.post2209072238 |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| torchaudio | 2.11.0+cu128 |
| numpy | 2.2.5 |
| pandas | 2.3.3 |
| scipy | 1.15.3 |
| opencv-python | 4.13.0.92 |
| Pillow | 12.2.0 |
| matplotlib | 3.10.9 |
| seaborn | 0.13.2 |
| PyYAML | 6.0.3 |
| tqdm | 4.68.3 |
| psutil | 7.2.2 |
| einops | 0.8.2 |
| safetensors | 0.8.0 |
| timm | 1.0.27 |
| huggingface_hub | 1.21.0 |
| requests | 2.34.2 |
| filelock | 3.29.0 |
| fsspec | 2026.4.0 |
| sympy | 1.14.0 |
| networkx | 3.4.2 |
| typing_extensions | 4.15.0 |
| nvidia-ml-py | 13.610.43 |

See `requirements.txt` for the full pinned list.

## Dataset

The dataset configuration is expected at `../aortic_valve_colab.yaml` (relative to `train.py`):

```yaml
train: [
  "./datasets/train/images",
  # "./datasets/aug_mosaic/images",
  # "./datasets/aug_hsv/images",
  # "./datasets/aug_scale/images"
]
val: "./datasets/val/images"
test: "./datasets/test/images"

names:
  0: aortic_valve
```

> Dataset images are single-channel grayscale cardiac CT slices, single class (`aortic_valve`). Paths are relative to the location of this YAML file. The commented-out entries under `train` are optional offline-augmented image folders (mosaic / HSV / scale) that can be enabled by uncommenting them. Update the paths to match your local dataset location before training.

## Training

```bash
python train.py
```

Key training settings used in the paper (see `train.py`):

| Setting | Value |
|---|---|
| Epochs | 100 |
| Image size | 640 |
| Batch size | 16 |
| Optimizer | SGD |
| Initial LR (`lr0`) | 0.001 |
| Weight decay | 0.0005 |
| Momentum | 0.937 |
| Pool size (attention) | 14 |

Training runs are saved under `runs/detect/<name>/`, with the best checkpoint at `runs/detect/<name>/weights/best.pt`.

### Three-Stage Progressive Training Schedule

The distance-metric ablation models and the final LMA-YOLO model were trained using a three-stage progressive schedule. Patience is the early-stopping patience (epochs); Stage I uses no early stopping.

| Stage | Initial weights | lr0 | Epochs | Patience |
|---|---|---|---|---|
| Stage I | yolo12n.pt | 0.001 | 100 | – |
| Stage II | Stage-I best.pt | 0.0002 | 30 | 10 |
| Stage III | Stage-II best.pt | 0.0001 | 20 | 10 |

Each stage resumes from the previous stage's best checkpoint (`best.pt`) with a reduced learning rate, allowing the model to progressively fine-tune without overfitting.

## Model Configuration

`yolov12_LMA-YOLO.yaml` defines the LMA-YOLO architecture and contains the standard `scales:` dictionary (n/s/m/l/x) inherited from YOLOv12. Since the file itself has no scale suffix, `train.py` loads it with a scale-prefixed string (`YOLO("yolov12n_LMA-YOLO.yaml")`), which tells Ultralytics to load `yolov12_LMA-YOLO.yaml` using the `n` (nano) entry from `scales:`. This is standard Ultralytics behavior, not a separate file.

- **Backbone**: standard YOLOv12-turbo backbone (Conv + C3k2 + A2C2f blocks).
- **Head**: `A2C2fManhattan` replaces the P4-level attention block, using pure Manhattan (L1) distance instead of the mixed Manhattan–Cosine similarity used in prior work, followed by standard `A2C2f` / `C3k2` blocks at P3 and P5.
- **Detect**: multi-scale detection head over P3, P4, P5.

## Evaluation

Validation/test-set evaluation follows the standard Ultralytics workflow, e.g.:

```bash
yolo val model=runs/detect/pool_size_14/weights/best.pt data=../aortic_valve_colab.yaml split=test imgsz=640
```

Reported test-set results (final model, three-stage training):

| Model | Params (M) | GFLOPs | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| YOLOv12 (baseline) | 2.51 | 5.8 | 0.921 | 0.917 | 0.954 | 0.623 |
| **LMA-YOLO (Ours)** | 2.58 | 6.0 | 0.923 | 0.948 | 0.961 | 0.637 |

## SOTA Comparisons

LMA-YOLO is compared against the following state-of-the-art lightweight detectors (retrained on the same dataset):

- [RCS-YOLO](https://github.com/mkang315/RCS-YOLO)
- [PK-YOLO](https://github.com/mkang315/PK-YOLO)

## License

This project builds on [Ultralytics YOLOv12](https://github.com/sunsmarterjie/yolov12) and follows its **AGPL-3.0** license.

## Acknowledgements

- Base architecture: [YOLOv12](https://github.com/sunsmarterjie/yolov12) (Tian et al., 2025)
- Attention design inspired by SMA-YOLO (Guo et al., 2025)
