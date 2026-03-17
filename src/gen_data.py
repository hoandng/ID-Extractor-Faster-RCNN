"""
gen_data.py
Dung model da train de tu dong tao dataset cho model tiep theo.

Luong:
    Card model da train
        python gen_data.py configs/gen_card.yaml
    -> Crop + padding -> luu vao dataset_corner/

    Corner model da train
        python gen_data.py configs/gen_corner.yaml
    -> Warp perspective -> luu vao dataset_field/
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import torch
import yaml

from model import load_model
from utils import add_padding, preprocess_image, warp_perspective

CORNER_LABEL_TO_NAME = {
    1: "top_left", 2: "top_right",
    3: "bottom_right", 4: "bottom_left",
}


@torch.no_grad()
def detect(model, img_bgr, device):
    tensor = preprocess_image(img_bgr).to(device)
    return model([tensor])[0]


def best_box(pred, score_thresh):
    scores = pred["scores"].cpu()
    if len(scores) == 0 or scores.max().item() < score_thresh:
        return None
    return pred["boxes"][scores.argmax()].cpu().numpy().astype(int)


def best_corners(pred, score_thresh):
    boxes  = pred["boxes"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    best   = {}
    for box, lbl, sc in zip(boxes, labels, scores):
        lbl = int(lbl)
        if lbl not in CORNER_LABEL_TO_NAME or sc < score_thresh:
            continue
        if lbl not in best or sc > best[lbl][0]:
            best[lbl] = (sc, box)
    return {CORNER_LABEL_TO_NAME[lbl]: ((b[0]+b[2])/2, (b[1]+b[3])/2)
            for lbl, (_, b) in best.items()}


# def save_pair(img, stem, out_img_dir, src_xml, out_ann_dir):
#     cv2.imwrite(str(Path(out_img_dir) / f"{stem}.jpg"), img)
#     shutil.copy2(src_xml, Path(out_ann_dir) / f"{stem}.xml")


def generate_corner_dataset(cfg, device):
    print("\n[Card -> Corner] Tao dataset corner...")
    model = load_model(cfg["card_weights"], num_classes=2, device=device,
                       score_thresh=cfg.get("score_thresh", 0.4))

    src_img = Path(cfg["img_dir"])
    # src_ann = Path(cfg["ann_dir"])
    out_img = Path(cfg["out_img_dir"])
    # out_ann = Path(cfg["out_ann_dir"])
    out_img.mkdir(parents=True, exist_ok=True)
    # out_ann.mkdir(parents=True, exist_ok=True)

    ok = skip = 0
    for img_path in sorted(src_img.glob("*.[jp][pn]g")):
        img    = cv2.imread(str(img_path))
        padded = add_padding(img, pad_size= 50, mode="pixel")
        pred   = detect(model, padded, device)
        box    = best_box(pred, cfg.get("score_thresh", 0.4))

        if box is None:
            print(f"  x Khong detect duoc the: {img_path.name}")
            skip += 1; continue

        x1, y1, x2, y2 = box
        h, w = padded.shape[:2]
        cropped = padded[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        # xml_path = src_ann / f"{img_path.stem}.xml"

        # if cropped.size == 0 or not xml_path.exists():
        #     skip += 1; continue

        # save_pair(cropped, img_path.stem, out_img, xml_path, out_ann)
        cv2.imwrite(str(Path(out_img) / f"{img_path.stem}.jpg"), cropped)
        ok += 1

    print(f"  Hoan thanh: {ok} thanh cong, {skip} bo qua")
    print(f"  Dataset corner: {out_img.parent}")


def generate_field_dataset(cfg, device):
    print("\n[Corner -> Field] Tao dataset field...")
    model = load_model(cfg["corner_weights"], num_classes=5, device=device,
                       score_thresh=cfg.get("score_thresh", 0.35))

    src_img = Path(cfg["img_dir"])
    # src_ann = Path(cfg["ann_dir"])
    out_img = Path(cfg["out_img_dir"])
    # out_ann = Path(cfg["out_ann_dir"])
    out_img.mkdir(parents=True, exist_ok=True)
    # out_ann.mkdir(parents=True, exist_ok=True)

    ok = skip = 0
    for img_path in sorted(src_img.glob("*.[jp][pn]g")):
        img     = cv2.imread(str(img_path))
        pred    = detect(model, img, device)
        corners = best_corners(pred, cfg.get("score_thresh", 0.35))

        if len(corners) < 4:
            missing = [k for k in CORNER_LABEL_TO_NAME.values()
                       if k not in corners]
            print(f"  x Thieu goc {missing}: {img_path.name}")
            skip += 1; continue

        warped   = warp_perspective(img, corners)
        # xml_path = src_ann / f"{img_path.stem}.xml"

        # if warped is None or not xml_path.exists():
        #     skip += 1; continue

        # save_pair(warped, img_path.stem, out_img, xml_path, out_ann)
        cv2.imwrite(str(Path(out_img) / f"{img_path.stem}.jpg"), warped)
        ok += 1

    print(f"  Hoan thanh: {ok} thanh cong, {skip} bo qua")
    print(f"  Dataset field: {out_img.parent}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args   = parser.parse_args()
    cfg    = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    step = cfg.get("gen_step")
    if step == "card_to_corner":
        generate_corner_dataset(cfg, device)
        print("\n-> Buoc tiep theo: python train.py configs/corner.yaml")
    elif step == "corner_to_field":
        generate_field_dataset(cfg, device)
        print("\n-> Buoc tiep theo: python train.py configs/field.yaml")
    else:
        print(f"gen_step khong hop le: '{step}'")
        print("Can: 'card_to_corner' hoac 'corner_to_field'")


if __name__ == "__main__":
    main()