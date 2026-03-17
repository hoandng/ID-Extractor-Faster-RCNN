"""
inference.py
Pipeline 6 buoc trich xuat thong tin CCCD.

Cach dung:
    from inference import CCCDPipeline

    pipe   = CCCDPipeline("weights/card/best.pth",
                          "weights/corner/best.pth",
                          "weights/field/best.pth")
    result = pipe.run("anh_cccd.jpg")
    print(result)
    # {"status": "success", "data": {"hoten": "NGUYEN VAN A", ...}}
"""

import cv2
import torch
from PIL import Image

from src.model import load_model
from src.utils import add_padding, preprocess_image, warp_perspective

CORNER_LABELS = {
    1: "top_left", 2: "top_right",
    3: "bottom_right", 4: "bottom_left",
}
FIELD_LABELS = {
    1: "id",       2: "hoten",    3: "ngaysinh", 4: "gioitinh",
    5: "quoctich", 6: "quequan",  7: "diachi",   8: "giatriden",
}


class CCCDPipeline:

    def __init__(self, card_model, corner_model, field_model, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Loading models ({self.device})...")
        self.card_model   = load_model(card_model,   num_classes=2,
                                       device=self.device)
        self.corner_model = load_model(corner_model, num_classes=5,
                                       device=self.device,
                                       score_thresh=0.35)
        self.field_model  = load_model(field_model,  num_classes=9,
                                       device=self.device,
                                       score_thresh=0.35)
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor
        ocr_cfg = Cfg.load_config_from_name("vgg_seq2seq")
        ocr_cfg["device"] = self.device
        self.ocr = Predictor(ocr_cfg)
        print("San sang!\n")

    @torch.no_grad()
    def run(self, image_input):
        img = (cv2.imread(image_input)
               if isinstance(image_input, str)
               else image_input.copy())
        if img is None:
            return {"status": "error", "message": "Khong doc duoc anh"}

        # Buoc 1+2: Detect the -> Crop
        padded   = add_padding(img, pad_size= 50, mode="pixel")
        card_box = self._detect_card(padded)
        if card_box is None:
            return {"status": "no_card"}
        x1, y1, x2, y2 = card_box
        cropped = padded[y1:y2, x1:x2]
        cropped = add_padding(cropped, pad_size=50, mode="pixel")
        # Buoc 3+4: Detect goc -> Warp
        corners = self._detect_corners(cropped)
        if len(corners) < 4:
            return {"status": "corners_missing", "found": list(corners)}
        warped = warp_perspective(cropped, corners)
        if warped is None:
            return {"status": "warp_failed"}

        # Buoc 5: Detect truong thong tin
        fields = self._detect_fields(warped)
        if not fields:
            return {"status": "no_fields"}

        # Buoc 6: OCR
        data = self._run_ocr(warped, fields)
        return {"status": "success", "data": data}

    def _detect_card(self, img, thresh=0.4):
        tensor = preprocess_image(img).to(self.device)
        pred   = self.card_model([tensor])[0]
        scores = pred["scores"]
        if len(scores) == 0 or scores.max().item() < thresh:
            return None
        box = pred["boxes"][scores.argmax()].cpu().numpy().astype(int)
        return tuple(box)

    def _detect_corners(self, img, thresh=0.35):
        tensor = preprocess_image(img).to(self.device)
        pred   = self.corner_model([tensor])[0]
        boxes  = pred["boxes"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        best   = {}
        for box, lbl, sc in zip(boxes, labels, scores):
            lbl = int(lbl)
            if lbl not in CORNER_LABELS or sc < thresh:
                continue
            if lbl not in best or sc > best[lbl][0]:
                best[lbl] = (sc, box)
        return {CORNER_LABELS[lbl]: ((b[0]+b[2])/2, (b[1]+b[3])/2)
                for lbl, (_, b) in best.items()}

    def _detect_fields(self, img, thresh=0.35):
        tensor = preprocess_image(img).to(self.device)
        pred   = self.field_model([tensor])[0]
        boxes  = pred["boxes"].cpu().numpy().astype(int)
        labels = pred["labels"].cpu().numpy()
        h, w   = img.shape[:2]
        fields = {}
        for box, lbl in zip(boxes, labels):
            name = FIELD_LABELS.get(int(lbl))
            if name is None:
                continue
            x1 = max(0, box[0]); y1 = max(0, box[1])
            x2 = min(w, box[2]); y2 = min(h, box[3])
            if x2 > x1 and y2 > y1:
                fields.setdefault(name, []).append((x1, y1, x2, y2))
        for name in fields:
            fields[name].sort(key=lambda b: b[1])
        return fields

    def _run_ocr(self, img, fields):
        result = {}
        for name, boxes in fields.items():
            texts = []
            for x1, y1, x2, y2 in boxes:
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                pil  = Image.fromarray(
                    cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                )
                text = self.ocr.predict(pil).strip()
                if text:
                    texts.append(text)
            if texts:
                result[name] = " ".join(texts)
        return result