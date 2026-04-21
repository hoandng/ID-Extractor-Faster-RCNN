import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.utils import preprocess_image


def read_annotation(xml_path):
    root   = ET.parse(xml_path,
                      parser=ET.XMLParser(encoding="utf-8")).getroot()
    width  = int(float(root.find("size/width").text))
    height = int(float(root.find("size/height").text))

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bb   = obj.find("bndbox")
        xmin = max(0.0, float(bb.find("xmin").text))
        ymin = max(0.0, float(bb.find("ymin").text))
        xmax = min(float(bb.find("xmax").text), width  - 1.0)
        ymax = min(float(bb.find("ymax").text), height - 1.0)
        if xmax > xmin and ymax > ymin:
            objects.append({"name": name,
                            "bbox": [xmin, ymin, xmax, ymax]})
    return objects


class CCCDDataset(Dataset):
    def __init__(self, img_dir, ann_dir, class_map):
        self.class_map = class_map
        self.samples   = []

        for xml in sorted(Path(ann_dir).glob("*.xml")):
            for ext in [".jpg", ".jpeg", ".png"]:
                img = Path(img_dir) / f"{xml.stem}{ext}"
                if img.exists():
                    self.samples.append((img, xml))
                    break

        if not self.samples:
            raise FileNotFoundError(
                f"Khong tim thay anh nao!\n"
                f"  Anh: {img_dir}\n"
                f"  XML: {ann_dir}"
            )
        print(f"  Dataset: {len(self.samples)} mau")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, xml_path = self.samples[idx]

        img     = cv2.imread(str(img_path))
        objects = read_annotation(xml_path)

        boxes, labels = [], []
        for obj in objects:
            label = self.class_map.get(obj["name"])
            if label is not None:
                boxes.append(obj["bbox"])
                labels.append(label)

        if not boxes:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)

        return preprocess_image(img), {"boxes": boxes_t, "labels": labels_t}


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def make_dataloaders(img_dir, ann_dir, class_map,
                     batch_size=2, val_ratio=0.15):
    dataset = CCCDDataset(img_dir, ann_dir, class_map)
    indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(indices)

    n_val     = max(1, int(len(indices) * val_ratio))
    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]
    print(f"  Train: {len(train_idx)}  |  Val: {len(val_idx)}")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 2,
        collate_fn  = collate_fn,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size  = 1,
        shuffle     = False,
        num_workers = 2,
        collate_fn  = collate_fn,
    )
    return train_loader, val_loader