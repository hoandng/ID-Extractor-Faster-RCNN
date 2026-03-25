import json
import time
from pathlib import Path

import numpy as np
import torch
from torchmetrics import ConfusionMatrix
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou
from tqdm import tqdm

from src.visualize import plot_confusion_matrix


@torch.no_grad()
def full_evaluation(model, dataloader, device, cfg, save_dir, writer=None):
    """
    Tính và lưu toàn bộ metrics đánh giá:
        - mAP@0.5, mAP@0.5:0.95, mAR@100
        - Confusion Matrix  →  confusion_matrix.png
        - Inference Time (FPS, ms/ảnh)
        - evaluation_report.txt + .json
    """
    model.eval()
    save_dir    = Path(save_dir)
    class_names = list(cfg["class_map"].keys())
    num_classes = cfg["num_classes"]

    map_metric    = MeanAveragePrecision(iou_type="bbox")
    confmat       = ConfusionMatrix(task="multiclass", num_classes=num_classes)
    matched_preds = []
    matched_gts   = []
    times         = []

    # Warm up GPU
    warm_imgs, _ = next(iter(dataloader))
    warm_imgs    = [img.to(device) for img in warm_imgs]
    for _ in range(3):
        model(warm_imgs)

    for images, targets in tqdm(dataloader, desc="  Eval ", leave=False):
        images = [img.to(device) for img in images]

        # Đo inference time
        if device == "cuda":
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); preds = model(images); e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter()
            preds = model(images)
            times.append((time.perf_counter() - t0) * 1000)

        # mAP
        map_metric.update(
            [{"boxes":  p["boxes"].cpu(),
              "scores": p["scores"].cpu(),
              "labels": p["labels"].cpu()} for p in preds],
            [{"boxes":  t["boxes"].cpu(),
              "labels": t["labels"].cpu()} for t in targets],
        )

        # Confusion Matrix: ghép predicted label với gt label khớp nhất
        for pred, target in zip(preds, targets):
            pb = pred["boxes"].cpu()
            pl = pred["labels"].cpu()
            gb = target["boxes"]
            gl = target["labels"]
            if len(pb) == 0 or len(gb) == 0:
                continue
            iou = box_iou(pb, gb)
            for gi in range(len(gb)):
                best_iou, best_pi = iou[:, gi].max(0)
                if best_iou >= 0.5:
                    matched_preds.append(
                        pl[best_pi.item()].clamp(0, num_classes - 1))
                    matched_gts.append(
                        gl[gi].clamp(0, num_classes - 1))

    # Tổng hợp
    r      = map_metric.compute()
    map_50 = round(r["map_50"].item(), 4)
    map_   = round(r["map"].item(),    4)
    mar    = round(r["mar_100"].item(),4)
    avg_ms = round(sum(times) / len(times), 1) if times else 0
    fps    = round(1000 / avg_ms, 1)            if avg_ms > 0 else 0

    # Confusion Matrix PNG
    if matched_preds:
        confmat.update(torch.stack(matched_preds),
                       torch.stack(matched_gts))
        cm  = confmat.compute().numpy()
        fig = plot_confusion_matrix(
            cm, class_names,
            title=f"Confusion Matrix — {Path(cfg['weights_dir']).name}",
        )
        fig.savefig(save_dir / "confusion_matrix.png",
                    dpi=120, bbox_inches="tight")
        if writer:
            writer.add_figure("Eval/ConfusionMatrix", fig)
        import matplotlib.pyplot as plt
        plt.close(fig)
        print("  [OK] confusion_matrix.png")

    # TensorBoard
    if writer:
        writer.add_scalar("Eval/mAP@0.5",      map_50, 0)
        writer.add_scalar("Eval/mAP@0.5:0.95", map_,   0)
        writer.add_scalar("Eval/mAR@100",       mar,    0)
        writer.add_scalar("Eval/FPS",           fps,    0)

    report = {
        "mAP@0.5":      map_50,
        "mAP@0.5:0.95": map_,
        "mAR@100":      mar,
        "FPS":          fps,
        "ms_per_image": avg_ms,
    }
    _save_report(report, save_dir,
                 model_name=Path(cfg["weights_dir"]).name)
    return report


def _save_report(report, save_dir, model_name):
    lines = [
        "=" * 45,
        f"EVALUATION REPORT — {model_name.upper()}",
        "=" * 45,
        f"  mAP@0.5       : {report['mAP@0.5']}",
        f"  mAP@0.5:0.95  : {report['mAP@0.5:0.95']}",
        f"  mAR@100       : {report['mAR@100']}",
        "-" * 45,
        f"  FPS           : {report['FPS']}",
        f"  ms / image    : {report['ms_per_image']}",
        "=" * 45,
    ]
    save_dir = Path(save_dir)
    with open(save_dir / "evaluation_report.txt", "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(save_dir / "evaluation_report.json", "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("  [OK] evaluation_report.txt")
    for line in lines:
        print(line)