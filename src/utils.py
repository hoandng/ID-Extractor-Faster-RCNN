"""
utils.py
Cac ham xu ly anh dung chung cho train va inference.
"""

import cv2
import numpy as np
import torchvision.transforms.functional as TF


def preprocess_image(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor  = TF.to_tensor(img_rgb)
    tensor  = TF.normalize(tensor,
                           mean=[0.485, 0.456, 0.406],
                           std =[0.229, 0.224, 0.225])
    return tensor


def add_padding(img, pad_size=50, mode='pixel'):

    h, w = img.shape[:2]

    if mode == 'percent':
        top = bottom = int(h * pad_size)
        left = right = int(w * pad_size)
    else:
        top = bottom = left = right = int(pad_size)

    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_img


def warp_perspective(img, corners):
    required = ["top_left", "top_right", "bottom_right", "bottom_left"]
    if not all(k in corners for k in required):
        return None

    tl = np.array(corners["top_left"], dtype=np.float32)
    tr = np.array(corners["top_right"], dtype=np.float32)
    br = np.array(corners["bottom_right"], dtype=np.float32)
    bl = np.array(corners["bottom_left"], dtype=np.float32)

    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    w, h = max(w, 32), max(h, 32)

    if h > w:
        w, h    = h, w
        src_pts = np.array([bl, tl, tr, br], dtype=np.float32)
    else:
        src_pts = np.array([tl, tr, br, bl], dtype=np.float32)

    RATIO = 1.586
    if w / h > RATIO:
        h = int(w / RATIO)
    else:
        w = int(h * RATIO)

    dst_pts = np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (w, h), flags= cv2.INTER_CUBIC)