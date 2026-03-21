import numpy as np
import pandas as pd
from abc import abstractmethod
from core import Visualization, Object
from image import ImageDataset
from visualization import relationship

class ContrastAndBrightness(Visualization):

    def __init__(self):
        self._dataset: ImageDataset | None = None
        self._df: pd.DataFrame | None = None
        self._status: str = "Not Run"
    
    def run(self, obj: ImageDataset):
        if self._dataset is None:
            self._status = "Failed"
            return
        
        if isinstance(obj, ImageDataset):
            self.visitImageDataset(obj)
            relationship.visualize_brightness_contrast_boxplot(self._df)        

        return

    def log(self):
        print("=" * 50)
        print("Step        : Analysis Contrast & Brightness")
        print(f"Dataset     : {self._dataset._folder_path if self._dataset else 'None'}")
        print(f"Status      : {self._status}")
        if self._df is not None:
            print("\nResult summary (mean per class):")
            print(
                self._df.groupby("class_name")[["brightness", "contrast"]]
                .mean()
                .round(2)
                .to_string()
            )
        print("=" * 50)

    def visitImageDataset(self, obj: ImageDataset):
        """
        Task 15 

        Công thức:
            gray       = 0.299·R + 0.587·G + 0.114·B   (ITU-R BT.601)
            brightness = mean(gray)   ∈ [0, 255]
            contrast   = std(gray)    ∈ [0, 127.5]

        Lưu kết quả vào self._df (pd.DataFrame) với columns:
            ['class_name', 'brightness', 'contrast']
        """
        try:
            images, labels = obj.images                              # (N,H,W,3) uint8, (N,) int64
            idx_to_class   = {v: k for k, v in obj.class_idx.items()}  # {1:'dog', ...}

            records = []
            for img, label in zip(images, labels):
                gray = (
                    0.299 * img[:, :, 0].astype(np.float32)
                  + 0.587 * img[:, :, 1].astype(np.float32)
                  + 0.114 * img[:, :, 2].astype(np.float32)
                )
                records.append({
                    "class_name": idx_to_class.get(int(label), str(label)),
                    "brightness": float(np.mean(gray)),
                    "contrast"  : float(np.std(gray)),
                })

            self._df      = pd.DataFrame(records, columns=["class_name", "brightness", "contrast"])
            self._dataset = obj
            self._status  = "Success"

        except Exception as e:
            self._status = f"Failed — {e}"
