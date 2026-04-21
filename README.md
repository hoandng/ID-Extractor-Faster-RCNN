# Vietnamese ID Card Information Extractor

An automated system for extracting information from photos of **Vietnamese ID cards (12-digit version)** using Faster R-CNN + ResNet-18 + VietOCR.

---

## Table of Contents

- [Introduction](#introduction)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Web Interface](#web-interface)
- [Training Results](#training-results)

---

## Introduction

This project addresses the problem of manual data entry from ID cards—which is time-consuming and error-prone—by building an end-to-end pipeline that automatically recognizes and returns information in JSON format.

**Extracted Information:**

| Key | Information Field |
|-----|----------------|
| `id` | ID Number |
| `hoten` | Full Name |
| `ngaysinh` | Date of Birth |
| `gioitinh` | Gender |
| `quoctich` | Nationality |
| `quequan` | Place of Origin |
| `diachi` | Place of Residence |
| `giatriden` | Valid Until |

**Sample Result:**

```python
# {
#   "status": "success",
#   "data": {
#     "id":        "079123456789",
#     "hoten":     "NGUYEN VAN A",
#     "ngaysinh":  "01/01/1990",
#     "gioitinh":  "Nam",
#     "quoctich":  "Viet Nam",
#     "quequan":   "Ha Noi",
#     "diachi":    "So 1 Duong ABC Quan XYZ Ha Noi",
#     "giatriden": "01/01/2030"
#   }
# }
```

---

## System Architecture

The pipeline consists of **4 sequential stages**. The output of each stage is the input for the next:

```
           Input Image
                │
                ▼
┌───────────────────────────────┐
│  Stage 1 — Card Detector      │
│  Faster R-CNN, num_classes=2  │
│  Find the ID card region      │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  Stage 2 — Corner Detector    │
│  Faster R-CNN, num_classes=5  │
│  Identify the 4 card corners  │
└──────────────┬────────────────┘
               │
               │
               ▼
┌───────────────────────────────┐
│  Stage 3 — Perspective Warp   │
│  cv2.getPerspectiveTransform  │
│  Straighten the skewed image  │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  Stage 4 — Field Detector     │
│  Faster R-CNN, num_classes=9  │
│  Detect 8 information fields  │
│           +                   │
│  VietOCR (vgg_seq2seq)        │
│  Recognize Vietnamese text    │
└──────────────┬────────────────┘
               │
               ▼
           Result
```

---

## Project Structure

```
cccd-extractor/
├── src/                                # Core source code modules
│   ├── dataset.py                      # Custom Dataset and DataLoader implementation
│   ├── evaluate.py                     # Model performance evaluation scripts
│   ├── inference.py                    # Prediction pipeline for production/testing
│   ├── model.py                        # Deep Learning architecture definitions
│   ├── utils.py                        # Shared utility functions and helper methods
│   └── visualize.py                    # Tools for plotting and result visualization
│
├── configs/                            # Training and deployment configurations
│   ├── card.yaml                       # Hyperparameters for ID Card Detection
│   ├── corner.yaml                     # Hyperparameters for Corner Points Detection
│   ├── field.yaml                      # Hyperparameters for Information Field Detection
│   ├── gen_card.yaml                   # Pipeline config: Generating Corner dataset from Card model
│   └── gen_corner.yaml                 # Pipeline config: Generating Field dataset from Corner model
│
├── dataset/                            # Data storage organized by task
│   ├── card/                           # Full ID card detection data
│   │   ├── images/                     # Raw input images
│   │   └── annotations/                # Pascal VOC XML labels
│   ├── corner/                         # Similar structure to dataset/card/
│   └── field/                          # Similar structure to dataset/card/
│
├── weights/                            # Model checkpoints and training artifacts
│   ├── card/                           # Output for ID Card Detection task
│   │   ├── best.pth                    # Top-performing model weights
│   │   ├── last.pth                    # Most recent checkpoint
│   │   ├── results.png                 # Training metrics visualization
│   │   ├── tensorboard/                # Logs for TensorBoard monitoring
│   │   ├── confusion_matrix.png        # Error analysis matrix
│   │   └── evaluation_report/          # Detailed metrics (precision, recall, mAP) in TXT/JSON
│   ├── corner/                         # Similar structure to weights/card/
│   └── field/                          # Similar structure to weights/card/
│
├── train.py                            # Main script to initiate model training
├── gen_data.py                         # Automated dataset generation using pre-trained models
├── app.py                              # Web-based interface (Streamlit application)
├── main.py                             # Entry point for end-to-end inference pipeline
└── README.md                           # Project documentation and setup guide
```

---

## Installation

### Hardware Requirements

| Component | Minimum | Recommended |
|------------|-----------|-------------|
| GPU VRAM | 4 GB  | 8 GB+ |
| System RAM | 8 GB | 16 GB |
| Disk Space | 5 GB | 15 GB |
| Python | 3.8+ | 3.10+ |
| CUDA | 11.8 | 12.1+ |

### Step 1 — Install PyTorch with CUDA

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 2 — Install Remaining Libraries

```bash
pip install -r requirements.txt
```

---

## Data Preparation

### Annotation Format — Pascal VOC XML

Each image corresponds to an `.xml` file with the same name, placed in the `annotations/` directory:

```xml
<annotation>
  <filename>cccd_001.jpg</filename>
  <size>
    <width>1920</width>
    <height>1080</height>
  </size>
  <object>
    <name>cccd</name>
    <bndbox>
      <xmin>312</xmin>
      <ymin>245</ymin>
      <xmax>1608</xmax>
      <ymax>835</ymax>
    </bndbox>
  </object>
</annotation>
```

---

## Training

### Mandatory Training Order

```
Card Detector  →  Corner Detector  →  Field Detector
```

Each model requires its own dataset. The datasets for the corner and field detectors can be generated automatically from previously trained models.

### Create Dataset
```bash
python gen_data.py configs/gen_card.yaml
python gen_data.py configs/gen_corner.yaml
```

### Train Command

```bash
python train.py configs/card.yaml
python train.py configs/corner.yaml
python train.py configs/field.yaml
```

### Train with K-Fold Cross Validation

```bash
python train.py configs/card.yaml --kfold 5
```

After running, the `weights/card/` directory is automatically created:

```
weights/card/
├── fold_1/
│   ├── best.pth
│   ├── last.pth
│   └── results.png       ← separate graph for fold 1
├── fold_2/ … fold_5/
├── best.pth              ← fold with the highest mAP@0.5
└── kfold_results.png     ← composite 4-panel graph
```

### Resume from Interruption

```bash
python train.py configs/card.yaml --resume weights/card/last.pth
```

### Monitor with TensorBoard

```bash
tensorboard --logdir weights/card/tensorboard
```

Metrics are logged in real-time:

| Scalar | Meaning |
|--------|---------|
| `Loss/train` | Train loss per epoch |
| `Metrics/mAP@0.5` | mAP at IoU ≥ 0.5 |
| `Metrics/mAP@0.5:0.95` | Average mAP over 10 IoU thresholds |
| `Metrics/mAR@100` | Max recall with top-100 detections |
| `LR` | Current learning rate |
| `Compare/mAP` | Current mAP vs. best mAP (2 lines on the same chart) |

### `results.png` Graph — Automatically updated after each epoch

```
┌──────────────┬──────────────┬──────────────────┐
│  Train Loss  │   mAP@0.5   │  mAP@0.5:0.95   │
├──────────────┼──────────────┼──────────────────┤
│   mAR@100   │ Learning Rate│  mAP vs Best mAP │
└──────────────┴──────────────┴──────────────────┘
```

---

## Inference

Run:
```bash
python main.py
```

### Return Statuses

| `status` | Cause | Solution |
|----------|-------------|-----------|
| `success` | Success | Read `result["data"]` |
| `no_card` | Card not found | Retake the photo, ensuring the card is in the frame |
| `corners_missing` | Card corners missing | Card is obscured or the angle is too skewed |
| `warp_failed` | Warp failed | The four detected corners are invalid |
| `no_fields` | Fields not detected | The field model needs more training data |
| `error` | Image read error | Check the file path |

---

## Web Interface

```bash
streamlit run app.py
```

**"OCR Extraction" Page:**
- Upload ID card image (JPG, PNG)
- Click "Extract Information"
- Displays each field as a UI card
- Download results as JSON

**"Training Results" Page:**
- Display the status of each model (whether `best.pth` exists)
- View `results.png` (6-panel graph)
- View `kfold_results.png` if K-Fold was used
- View `kfold_summary.txt` in a code block
- Expand to view the `results.png` graph for each individual fold

---

## Training Results

After training is complete, each `weights/<model>/` directory contains:

| File | Description |
|------|-------|
| `best.pth` | Weights from the epoch with the highest mAP@0.5 |
| `last.pth` | Last checkpoint — used for resuming |
| `results.png` | 6-panel graph, overwritten after each epoch |
| `kfold_results.png` | Composite K-Fold graph (only when using `--kfold`) |
| `confusion_matrix.png` | Confusion matrix, generated after training is finished |
| `evaluation_report.txt` | mAP, mAR, FPS in text format |
| `evaluation_report.json` | mAP, mAR, FPS in JSON format |
| `tensorboard/` | Logs for TensorBoard |

**Sample `evaluation_report.txt`:**
```
=============================================
EVALUATION REPORT — CARD
=============================================
  mAP@0.5       : 0.9245
  mAP@0.5:0.95  : 0.7812
  mAR@100       : 0.9431
---------------------------------------------
  FPS           : 14.2
  ms / image    : 70.4
=============================================
```
## Common Error Handling

**`CUDA out of memory` during training:**
```yaml
# In the config yaml file, reduce:
batch_size: 1
max_size:   600
```

**Training interrupted mid-process:**
```bash
python train.py configs/card.yaml --resume weights/card/last.pth
```

**`gen_data.py` skips many images (high skip rate):**
Reduce `score_thresh` in the generation config file or improve the quality of the original dataset.

**OCR outputs text in the wrong order:**
The card was photographed at a sharp angle, causing the Corner Detector to detect corners in the wrong order. Ensure the 4 corners of the card are clearly visible in the image.
