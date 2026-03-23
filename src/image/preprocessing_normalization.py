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
        self._stats = Dict[str, Any] = {}
        return
    
    @property
    def stats(self):
        return
        
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
    
    def visitImageDataset():
        return

    def log(self):
        return
    
    def _fit_minmax(self, arr: np.ndarray):
        return
    
    def _fit_zscore_global(self, arr: np.ndarray):
        return
    
    def _fit_zscore_channel(self, arr: np.ndarray):
        return
    
    def _transform_minmax_01(self, arr: np.ndarray):
        return
    
    def _transform_minmax_m11(self, arr: np.ndarray):
        return

    def _transform_zscore_global(self, arr: np.ndarray):
        return
    
    def _transform_zscore_channel(self, arr: np.ndarray):
        return

