from image import ImageDataset
from core import Preprocessing
import numpy as np
from typing import Any, Dict
from config import SUPPORT_NORMALIZATION_METHOD,DEFAULT_NORMALIZATION_METHOD, DEFAULT_EPSILON
import cv2
import os
import time
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA

class Normalization(Preprocessing):
    def __init__(self,
                 method: str = DEFAULT_NORMALIZATION_METHOD,
                 eps: float = DEFAULT_EPSILON
                 ):
        """Khởi tạo lớp chuẩn hóa dữ liệu ảnh."""
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
        """Tính toán các thông số thống kê cần thiết."""
        if self._method in ("minmax_01", "minmax_m11"):
            self._fit_minmax(arr)
        elif self._method == "zscore_global":
            self._fit_zscore_global(arr)
        elif self._method == "zscore_channel":
            self._fit_zscore_channel(arr)
    
    def transform(self, arr: np.ndarray) -> np.ndarray:
        """Áp dụng công thức chuẩn hóa lên dữ liệu."""
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
        """Thực hiện cả tính toán thông số và chuẩn hóa."""
        self.fit(arr)
        return self.transform(arr)

    def run(self, obj: ImageDataset):
        """Thực thi quy trình chuẩn hóa trên tập dữ liệu."""
        if isinstance(obj, ImageDataset):
            self.visitImageDataset(obj)
        return

    def visitImageDataset(self, obj: ImageDataset):
        """Xử lý chuẩn hóa cụ thể cho ImageDataset."""
        try:
            images, labels = obj.images
            normalized_images = self.fit_transform(images)
            obj._images = (normalized_images, labels)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.log()
        return

    def log(self):
        """In thông số thống kê đã tính toán."""
        print(f"Method: {self._method}")
        for key, value in self._stats.items():
            if isinstance(value, np.ndarray):
                formatted_val = np.round(value, 4).tolist()
                print(f"  - {key}: {formatted_val}")
            else:
                print(f"  - {key}: {value:.4f}")
        return
    
    def _fit_minmax(self, arr: np.ndarray):
        """Tính Min và Max của mảng dữ liệu."""
        arr_float = arr.astype(np.float32)
        self._stats['min'] = np.min(arr_float)
        self._stats['max'] = np.max(arr_float)
        return
    
    def _fit_zscore_global(self, arr: np.ndarray):
        """Tính Mean và STD toàn cục."""
        arr_float = arr.astype(np.float32)
        self._stats['mean'] = np.mean(arr_float)
        self._stats['std'] = np.std(arr_float)
        return
    
    def _fit_zscore_channel(self, arr: np.ndarray):
        """Tính Mean và STD cho từng kênh màu."""
        arr_float = arr.astype(np.float32)
        axes = tuple(range(arr_float.ndim - 1))
        self._stats['mean'] = np.mean(arr_float, axis=axes)
        self._stats['std'] = np.std(arr_float, axis=axes)
        return
    
    def _transform_minmax_01(self, arr: np.ndarray):
        """
        Nén dữ liệu về khoảng [0, 1].

        Input:
            arr: Mảng numpy.

        Output:
            Mảng numpy sau khi nén.
        """
        # công thức: (x - min) / (max - min)
        arr_float = arr.astype(np.float32)
        min_val = self._stats['min']
        max_val = self._stats['max']
        return (arr_float - min_val) / (max_val - min_val + self._eps)
    
    def _transform_minmax_m11(self, arr: np.ndarray):
        """Nén dữ liệu về khoảng [-1, 1]."""
        arr_float = arr.astype(np.float32)
        min_val = self._stats['min']
        max_val = self._stats['max']
        return 2 * (arr_float - min_val) / (max_val - min_val + self._eps) - 1

    def _transform_zscore_global(self, arr: np.ndarray):
        """Chuẩn hóa Z-score toàn cục."""
        arr_float = arr.astype(np.float32)
        mean = self._stats['mean']
        std = self._stats['std']
        return (arr_float - mean) / (std + self._eps)
    
    def _transform_zscore_channel(self, arr: np.ndarray):
        """Chuẩn hóa Z-score theo kênh màu."""
        arr_float = arr.astype(np.float32)
        mean = self._stats['mean']
        std = self._stats['std']
        return (arr_float - mean) / (std + self._eps)

class NormalizationEvaluator(Normalization):
    """
    Class kế thừa từ Normalization gốc hỗ trợ đánh giá mô hình.
    """
    def __init__(self, method: str = DEFAULT_NORMALIZATION_METHOD, eps: float = DEFAULT_EPSILON):
        """Khởi tạo bộ đánh giá chuẩn hóa tối ưu bộ nhớ."""
        super().__init__(method=method, eps=eps)
        self._transformed_data_cache = None 
        self._raw_uint8_cache = None

    def visitImageDataset(self, obj: ImageDataset):
        """Xử lý chuẩn hóa an toàn bộ nhớ bằng kỹ thuật chunking."""
        try:
            print(f"   [INFO] Kích hoạt Chế độ Chống Tràn RAM (Toán học Chunking)...")
            total_images = len(obj._file_names)
            
            arr_uint8 = np.empty((total_images, 128, 128, 3), dtype=np.uint8)
            current_idx = 0
            for batch, _ in obj.load():
                for img in batch:
                    arr_uint8[current_idx] = cv2.resize(img, (128, 128))
                    current_idx += 1
            
            print(f"   [INFO] Đang tính toán Stats siêu nhẹ (Né OOM 10GB)...")
            chunk_size = 1000
            if self._method in ("minmax_01", "minmax_m11"):
                self._stats['min'] = float(arr_uint8.min())
                self._stats['max'] = float(arr_uint8.max())
            elif self._method == "zscore_global":
                t_sum, t_sq_sum = 0.0, 0.0
                n_pixels = arr_uint8.size
                for i in range(0, total_images, chunk_size):
                    chunk = arr_uint8[i:i+chunk_size].astype(np.float64)
                    t_sum += chunk.sum()
                    t_sq_sum += (chunk**2).sum()
                mean_val = t_sum / n_pixels
                var_val = (t_sq_sum / n_pixels) - (mean_val**2)
                self._stats['mean'] = mean_val
                self._stats['std'] = np.sqrt(var_val)
            elif self._method == "zscore_channel":
                t_sum = np.zeros(3, dtype=np.float64)
                t_sq_sum = np.zeros(3, dtype=np.float64)
                n_pixels = total_images * 128 * 128
                for i in range(0, total_images, chunk_size):
                    chunk = arr_uint8[i:i+chunk_size].astype(np.float64)
                    t_sum += chunk.sum(axis=(0,1,2))
                    t_sq_sum += (chunk**2).sum(axis=(0,1,2))
                mean_val = t_sum / n_pixels
                var_val = (t_sq_sum / n_pixels) - (mean_val**2)
                self._stats['mean'] = mean_val.astype(np.float32)
                self._stats['std'] = np.sqrt(var_val).astype(np.float32)

            print(f"   [INFO] Băm nhỏ dữ liệu để đưa qua Transform -> PCA...")
            pca = IncrementalPCA(n_components=50, batch_size=chunk_size)

            for i in range(0, total_images, chunk_size):
                chunk = arr_uint8[i:i+chunk_size]
                chunk_norm = self.transform(chunk)
                pca.partial_fit(chunk_norm.reshape(chunk_norm.shape[0], -1))

            X_pca = np.empty((total_images, 50), dtype=np.float32)
            for i in range(0, total_images, chunk_size):
                chunk = arr_uint8[i:i+chunk_size]
                chunk_norm = self.transform(chunk)
                X_pca[i:i+chunk_size] = pca.transform(chunk_norm.reshape(chunk_norm.shape[0], -1))

            self._transformed_data_cache = X_pca
            self._raw_uint8_cache = arr_uint8
            self._labels = obj._labels
            self._file_names = obj._file_names
            self._class_idx = obj.class_idx
            
            obj._images = ("Data stored safely", self._labels)
            self._status = "Success"
        except Exception as e:
            self._status = f"Failed ({str(e)})"
            print(self._status)
        finally:
            self.log()
        return

    def save_images(self, base_dir: str = "../data/preprocessing/normalization"):
        """Lưu các ảnh đã chuẩn hóa ra ổ cứng phục vụ so sánh."""
        if self._raw_uint8_cache is None:
            print("   [ERROR] Không có dữ liệu để lưu.")
            return

        save_dir = os.path.join(base_dir, self._method)
        print(f"   [INFO] Đang lưu từng ảnh vật lý ra: {save_dir} ...")
        
        idx_to_class = {v: k for k, v in self._class_idx.items()}
        
        for idx in range(len(self._raw_uint8_cache)):
            img_raw = self._raw_uint8_cache[idx:idx+1]
            img_norm = self.transform(img_raw)[0]
            
            img_min, img_max = img_norm.min(), img_norm.max()
            img_disp = (img_norm - img_min) / (img_max - img_min + 1e-8) * 255.0
            img_disp = img_disp.astype(np.uint8)
            
            label, fname = self._labels[idx], self._file_names[idx]
            class_name = idx_to_class[label]
            class_dir = os.path.join(save_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            cv2.imwrite(os.path.join(class_dir, fname), img_disp)
            
        print(f"   [SUCCESS] Đã xuất xong thư mục {self._method}!")

    def evaluation(self, n_repeats: int = 3, max_epochs: int = 30):
        """Đánh giá hiệu quả chuẩn hóa bằng SGD Logistic Regression."""
        print(f"\n[EVALUATION] Train SGD Logistic Regression ({self._method})...")
        start_time = time.time()
        
        X = self._transformed_data_cache
        y = np.array(self._labels)
        classes = np.unique(y)

        metrics_history = []
        learning_curves = [] 
        
        for i in range(n_repeats):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i*42)
            
            model = SGDClassifier(
                loss='log_loss', 
                learning_rate='adaptive', 
                eta0=0.01,
                early_stopping=False,
                random_state=i*42, 
                n_jobs=-1
            )
            
            epoch_accs = []
            for epoch in range(max_epochs):
                model.partial_fit(X_train, y_train, classes=classes)
                acc = model.score(X_test, y_test)
                epoch_accs.append(acc)
            
            y_pred = model.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            metrics_history.append(metrics)
            learning_curves.append(epoch_accs)
            
        avg_metrics = {k: np.mean([m[k] for m in metrics_history]) for k in metrics_history[0].keys()}
        avg_curve = np.mean(learning_curves, axis=0).tolist()
        
        print(f"[RESULT] {self._method} (Mất {time.time() - start_time:.1f}s) - F1: {avg_metrics['f1_score']:.4f}\n")
        
        return avg_metrics, avg_curve