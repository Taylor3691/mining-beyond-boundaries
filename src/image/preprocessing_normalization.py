from image import ImageDataset
from core import Preprocessing
import numpy as np
from typing import Any, Dict
from config import SUPPORT_NORMALIZATION_METHOD,DEFAULT_NORMALIZATION_METHOD, DEFAULT_EPSILON

# Task 29
# Các hàm fit, transform, fit_transform không được sửa nữa
# Triển khai tất cả các hàm cần triển khai ở dưới
# Hàm visitImageDataset thực hiện chuẩn hóa theo method được truyền vào, sau khi chuẩn hóa xong thì gán lại
# Hàm log thông báo thông tin của step giống các task trước
# Mong muốn làm thêm 1 hàm mà có thể visualize được ma trận trước và sau khi chuẩn hóa
# Chỉ cần lấy 1 ảnh bất kì trong dataset (Ưu tiên size nhỏ hoặc resize luôn) rồi define 1 hàm bên comparision.py
# Hàm visualize nhận vào 2 ma trận trước và sau khi normalize, phải hiện giá trị số tại các ô

class Normalization(Preprocessing):
    def __init__(self,
                  method: str = DEFAULT_NORMALIZATION_METHOD,
                  eps: float = DEFAULT_EPSILON
                  ):
        if method not in SUPPORT_NORMALIZATION_METHOD:
            raise ValueError("Method not support")
        
        self._method = method
        self._eps = eps
        self._stats : Dict[str, Any] = {}
        return
    
    @property
    def stats(self):
        return self._stats
        
    def fit(self, arr: np.ndarray):
        if self._method in ("minmax_01", "minmax_m11"):
            self._fit_minmax(arr)
        elif self._method == "zscore_global":
            self._fit_zscore_global(arr)
        elif self._method == "zscore_channel":
            self._fit_zscore_channel(arr)
    
    def transform(self, arr: np.ndarray) -> np.ndarray:
        if self._method == "minmax_01":
            return self._transform_minmax_01(arr)
        if self._method == "minmax_m11":
            return self._transform_minmax_m11(arr)
        if self._method == "zscore_global":
            return self._transform_zscore_global(arr)
        if self._method == "zscore_channel":
            return self._transform_zscore_channel(arr)
        raise ValueError(f"Unsupported method: {self._method}")
    
    def fit_transform(self, arr: np.ndarray):
        self.fit(arr)
        return self.transform(arr)

    def run(self, obj: ImageDataset):
        if isinstance(obj, ImageDataset):
            self.visitImageDataset(obj)
        return
    
    def visitImageDataset(self, obj: ImageDataset):
        try:
            images, labels = obj.images
            
            # chuẩn hóa ảnh
            normalized_images = self.fit_transform(images)
            
            # Gán mảng đã chuẩn hóa vào đối tượng dataset
            obj._images = (normalized_images, labels)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.log()
        return

    def log(self):
        print(f"Method: {self._method}")
        
        for key, value in self._stats.items():
            if isinstance(value, np.ndarray):
                # Làm tròn để in cho đẹp, chuyển sang list để dễ nhìn
                formatted_val = np.round(value, 4).tolist()
                print(f"  - {key}: {formatted_val}")
            else:
                # In giá trị scalar (float)
                print(f"  - {key}: {value:.4f}")
        return
    
    def _fit_minmax(self, arr: np.ndarray):
        arr_float = arr.astype(np.float32)
        self._stats['min'] = np.min(arr_float)
        self._stats['max'] = np.max(arr_float)
        return
    
    def _fit_zscore_global(self, arr: np.ndarray):
        arr_float = arr.astype(np.float32)
        self._stats['mean'] = np.mean(arr_float)
        self._stats['std'] = np.std(arr_float)
        return
    
    def _fit_zscore_channel(self, arr: np.ndarray):
        # tính stats cho từng channel
        arr_float = arr.astype(np.float32)
        # lấy tất cả các trục trừ trục cuối cùng (channel), vd: (N,H,W,C), (H,W,C)
        axes = tuple(range(arr_float.ndim - 1))
        self._stats['mean'] = np.mean(arr_float, axis=axes)
        self._stats['std'] = np.std(arr_float, axis=axes)
        return
    
    def _transform_minmax_01(self, arr: np.ndarray):
        # công thức: (x - min) / (max - min)
        arr_float = arr.astype(np.float32)
        min_val = self._stats['min']
        max_val = self._stats['max']
        return (arr_float - min_val) / (max_val - min_val + self._eps)
    
    def _transform_minmax_m11(self, arr: np.ndarray):
        # công thức range bất kì : ((x - min) / (max - min)) * (max_range - min_range) - 1
        arr_float = arr.astype(np.float32)
        min_val = self._stats['min']
        max_val = self._stats['max']
        return 2 * (arr_float - min_val) / (max_val - min_val + self._eps) - 1

    def _transform_zscore_global(self, arr: np.ndarray):
        # công thức: (x - mean) / std với mean, std tính trên toàn bộ ảnh
        arr_float = arr.astype(np.float32)
        mean = self._stats['mean']
        std = self._stats['std']
        return (arr_float - mean) / (std + self._eps)
    
    def _transform_zscore_channel(self, arr: np.ndarray):
        # công thức: (x - mean) / std với mean, std tính trên từng channel
        arr_float = arr.astype(np.float32)
        mean = self._stats['mean']
        std = self._stats['std']
        return (arr_float - mean) / (std + self._eps)
    
    
# maybe log scale, decimal scale
