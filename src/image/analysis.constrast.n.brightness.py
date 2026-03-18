import numpy as np
import pandas as pd
from dataset import ImageDataset
from config import CLASS_INDEX

def compute_brightness_contrast_per_class(dataset: ImageDataset) -> pd.DataFrame:
    """
    Tính độ sáng (mean intensity) và độ tương phản (std intensity)
    theo từng lớp từ ImageDataset.

    Công thức:
        - Brightness = mean của toàn bộ pixel trên ảnh grayscale
          (dùng ITU-R BT.601: 0.299R + 0.587G + 0.114B)
        - Contrast   = std  của toàn bộ pixel trên ảnh grayscale
    -------
    pd.DataFrame với columns: ['class_name', 'brightness', 'contrast']
        - brightness : float, mean intensity [0.0, 255.0]
        - contrast   : float, std  intensity [0.0, 127.5]
    """
    images, labels = dataset.images                      # X: (N,H,W,3) uint8, Y: (N,) int64
    idx_to_class   = {v: k for k, v in dataset.class_idx.items()}  # {1:'dog', ...}

    records = []
    for img, label in zip(images, labels):
        # Chuyển RGB → Grayscale theo công thức ITU-R BT.601
        gray = (0.299 * img[:, :, 0].astype(np.float32)
              + 0.587 * img[:, :, 1].astype(np.float32)
              + 0.114 * img[:, :, 2].astype(np.float32))  # shape (H, W)

        brightness = float(np.mean(gray))
        contrast   = float(np.std(gray))
        class_name = idx_to_class.get(int(label), str(label))

        records.append({
            "class_name": class_name,
            "brightness": brightness,
            "contrast"  : contrast,
        })

    df = pd.DataFrame(records, columns=["class_name", "brightness", "contrast"])
    return df