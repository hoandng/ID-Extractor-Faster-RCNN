# CCCD Extractor — Hướng dẫn từng bước

## Cấu trúc project

```
utils.py        xử lý ảnh (preprocess, padding, warp)
dataset.py      đọc Pascal VOC, tạo DataLoader
model.py        tạo / load Faster R-CNN + ResNet-18
train.py        vòng lặp train
gen_data.py     dùng model đã train để tạo dataset cho model tiếp theo
inference.py    pipeline 6 bước đầy đủ

configs/
  card.yaml       config train card model
  gen_card.yaml   config tạo dataset corner
  corner.yaml     config train corner model
  gen_corner.yaml config tạo dataset field
  field.yaml      config train field model
```

## Cài đặt

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python albumentations vietocr tqdm pyyaml
```

## Luồng làm việc toàn bộ

```
dataset_raw/          (ảnh gốc + annotation card)
      │
      ▼
python train.py configs/card.yaml
      │
      ▼  weights/card/best.pth
      │
python gen_data.py configs/gen_card.yaml
      │  (crop thẻ → lưu vào dataset_corner/)
      ▼
python train.py configs/corner.yaml
      │
      ▼  weights/corner/best.pth
      │
python gen_data.py configs/gen_corner.yaml
      │  (warp ảnh → lưu vào dataset_field/)
      ▼
python train.py configs/field.yaml
      │
      ▼  weights/field/best.pth
      │
python inference.py
```

## Bước 1 — Chuẩn bị dataset gốc

Đặt ảnh và annotation Pascal VOC vào:
```
dataset_raw/
  images/       ← file ảnh (.jpg, .png)
  annotations/  ← file XML cùng tên với ảnh
```

Format XML mẫu:
```xml
<annotation>
  <filename>img001.jpg</filename>
  <size>
    <width>1920</width>
    <height>1080</height>
  </size>
  <object>
    <n>cccd</n>
    <difficult>0</difficult>
    <bndbox>
      <xmin>100</xmin><ymin>50</ymin>
      <xmax>900</xmax><ymax>580</ymax>
    </bndbox>
  </object>
</annotation>
```

## Bước 2 — Train Card model

```bash
python train.py configs/card.yaml
```

Weights lưu tại: `weights/card/best.pth`

## Bước 3 — Tạo dataset Corner (tự động)

```bash
python gen_data.py configs/gen_card.yaml
```

Script này sẽ:
- Chạy card model trên từng ảnh gốc
- Crop vùng thẻ ra
- Lưu vào `dataset_corner/images/` và `dataset_corner/annotations/`

## Bước 4 — Train Corner model

```bash
python train.py configs/corner.yaml
```

Weights lưu tại: `weights/corner/best.pth`

## Bước 5 — Tạo dataset Field (tự động)

```bash
python gen_data.py configs/gen_corner.yaml
```

Script này sẽ:
- Chạy corner model để tìm 4 góc
- Warp perspective về ảnh thẳng
- Lưu vào `dataset_field/images/` và `dataset_field/annotations/`

## Bước 6 — Train Field model

```bash
python train.py configs/field.yaml
```

## Chạy inference

```python
from src.inference import CCCDPipeline

pipe = CCCDPipeline(
    card_weights="weights/card/best.pth",
    corner_weights="weights/corner/best.pth",
    field_weights="weights/field/best.pth",
)
result = pipe.run("anh_cccd.jpg")

if result["status"] == "success":
    for field, text in result["data"].items():
        print(f"{field}: {text}")
```

## Nếu hết VRAM

Sửa trong file config:
```yaml
batch_size: 1
max_size:   640
```

## Resume nếu bị ngắt giữa chừng

```bash
python train.py configs/card.yaml --resume weights/card/last.pth
```
