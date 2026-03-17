"""
model.py
--------
Tạo và load model Faster R-CNN với backbone ResNet-18.

Faster R-CNN hoạt động như thế nào?
  1. Backbone (ResNet-18): đọc ảnh, trích xuất feature map
  2. FPN: gộp feature map từ nhiều tầng → nhìn được cả vật lớn lẫn nhỏ
  3. RPN: quét feature map, đề xuất ~2000 vùng có thể chứa object
  4. ROI Pooling: crop feature map theo từng vùng đề xuất → tensor cố định
  5. Head: phân loại từng vùng + tinh chỉnh tọa độ bbox

Tại sao ResNet-18 thay vì ResNet-50 mặc định?
  ResNet-50: ~25M tham số, cần ~2.2 GB VRAM
  ResNet-18: ~12M tham số, cần ~1.4 GB VRAM
  Với bài CCCD (ảnh chuẩn, ít class), ResNet-18 là đủ.

Cách dùng:
    from model import build_model, load_model

    # Tạo model mới để train
    model = build_model(num_classes=2)

    # Load model đã train
    model = load_model("weights/card/best.pth", num_classes=2)
"""

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


def build_model(num_classes, max_size=800, score_thresh=0.4):
    """
    Tao Faster R-CNN voi backbone ResNet-18 + FPN.

    Args:
        num_classes:  So class cua ban + 1 (background).
                      Vi du: detect "cccd" -> num_classes = 2
        max_size:     Canh dai nhat sau resize. Giam xuong neu het VRAM.
        score_thresh: Chi giu detection co confidence > nguong nay.
    """

    # Backbone: ResNet-18 pretrained ImageNet + FPN
    # FPN (Feature Pyramid Network) giup detect duoc ca vat lon lan nho
    backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
        backbone_name    = "resnet18",
        weights          = "DEFAULT",
        trainable_layers = 3,      # unfreeze 3 layer cuoi de fine-tune
    )

    # Anchor generator: cac "khung thu" dat day dac tren anh
    # RPN se xem moi anchor co chua object khong
    anchor_gen = AnchorGenerator(
        sizes         = ((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios = ((0.5, 1.0, 2.0),) * 5,
    )

    # ROI Pooler: crop feature map theo vung de xuat, resize ve 7x7
    roi_pooler = MultiScaleRoIAlign(
        featmap_names  = ["0", "1", "2", "3"],
        output_size    = 7,
        sampling_ratio = 2,
    )

    return FasterRCNN(
        backbone               = backbone,
        num_classes            = num_classes,
        rpn_anchor_generator   = anchor_gen,
        box_roi_pool           = roi_pooler,
        min_size               = 600,
        max_size               = max_size,
        box_score_thresh       = score_thresh,
        box_nms_thresh         = 0.4,
        box_detections_per_img = 20,
    )


def load_model(weights_path, num_classes, device="cpu",
               max_size=800, score_thresh=0.4):
    """Load model da train tu file .pth."""
    model = build_model(num_classes, max_size, score_thresh)
    ckpt  = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    return model.to(device).eval()