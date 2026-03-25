import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


def build_model(num_classes, max_size=800, score_thresh=0.4):
    """
    Tao Faster R-CNN voi backbone ResNet-18 + FPN.
    """

    # Backbone: ResNet-18 pretrained ImageNet + FPN
    # FPN (Feature Pyramid Network) giup detect duoc ca vat lon lan nho
    backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
        backbone_name    = "resnet18",
        weights          = "DEFAULT",
        trainable_layers = 3,
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