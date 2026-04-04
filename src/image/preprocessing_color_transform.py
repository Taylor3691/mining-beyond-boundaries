from image import ImageDataset
from core import Preprocessing
from config import DEFAULT_N_COMPONENTS, SUPPORT_COLOR_SPACE, COLOR_MAP, DEFAULT_SIZE
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import time

class ColorTransform(Preprocessing):
    def __init__(self, method: str = None, n: int = DEFAULT_N_COMPONENTS):
        if method is None:
            raise ValueError("Cannot let the method empty")
        elif method not in SUPPORT_COLOR_SPACE:
            raise ValueError("This color space is not support")

        self._step_name = "Preprocessing: Color Space Transform"
        self._dataset_name = ""
        self._n = n
        # Khởi tạo PCA và Scaler
        self._scaler = StandardScaler()
        # Batch size lớn hơn để đọc lẹ hơn
        self._pca = IncrementalPCA(n_components=n, batch_size=1024) 
        self._method = method
        # Cache dùng để lưu thẳng dữ liệu đã giảm chiều vào RAM
        self._transformed_data_cache = None 
        return
    
    # Hàm bắt buộc của Base Class
    def fit(self, obj: ImageDataset):
        if self._transformed_data_cache is None:
            self.fit_transform(obj)
        return self

    # Hàm thực thi chính: Đọc ảnh -> Đổi màu -> PCA -> Lưu Cache RAM
    def fit_transform(self, obj: ImageDataset):
        print(f"   [INFO] Khởi động động cơ đọc đĩa tốc độ cao...")
        code = COLOR_MAP[self._method]
        
        # Bước 1: Tính toán Standard Scaler (1 vòng lặp ổ cứng)
        print("      -> Bước 1: Quét dữ liệu để chuẩn hóa (Scaler)...")
        for batch, _ in obj.load():
            images = []
            for img in batch:
                if code is not None:
                    img = cv2.cvtColor(img, code)
                img = cv2.resize(img, DEFAULT_SIZE)
                images.append(img.flatten())
            self._scaler.partial_fit(np.array(images))

        # Bước 2: Tính toán PCA (1 vòng lặp ổ cứng)
        print("      -> Bước 2: Quét dữ liệu để tìm thành phần chính (PCA)...")
        for batch, _ in obj.load():
            images = []
            for img in batch:
                if code is not None:
                    img = cv2.cvtColor(img, code)
                img = cv2.resize(img, DEFAULT_SIZE)
                images.append(img.flatten())
            scaled_images = self._scaler.transform(np.array(images))
            self._pca.partial_fit(scaled_images)

        # Bước 3: Áp dụng biến đổi và Ép vào RAM (1 vòng lặp ổ cứng)
        print("      -> Bước 3: Nén dữ liệu xuống 50 chiều và đẩy vào RAM...")
        result = []
        for batch, _ in obj.load():
            images = []
            for img in batch:
                if code is not None:
                    img = cv2.cvtColor(img, code)
                img = cv2.resize(img, DEFAULT_SIZE)
                images.append(img.flatten())
            
            scaled_images = self._scaler.transform(np.array(images))
            pca_images = self._pca.transform(scaled_images)
            result.append(pca_images)
        
        # Gộp tất cả các batch lại thành 1 mảng numpy duy nhất (Rất nhẹ, ~10MB)
        self._transformed_data_cache = np.vstack(result)
        return self._transformed_data_cache

    def transform(self, obj: ImageDataset):
        if self._transformed_data_cache is not None:
            return [self._transformed_data_cache]
        raise Exception("Vui lòng gọi fit_transform() trước khi gọi transform().")

    def log(self):
        print(f"Bước xử lý : {self._step_name}")
        print(f"Tập dữ liệu: {self._dataset_name}")
        print(f"Trạng thái : {self._status}")
        print(f"Không gian màu: {self._method}")
        print(f"Tổng phương sai giải thích theo PCA với k={self._n}: {self._explanied_variance_sum:.6f}")
        print(f"Tổng tỷ lệ phương sai giải thích: {self._explanied_variance_ratio_sum:.6f}")
            
    def evaluation(self, obj: ImageDataset, n_repeats: int = 3):
        print(f"\n[EVALUATION] Bắt đầu huấn luyện mô hình (Không gian: {self._method})...")
        start_time = time.time()
        
        if self._transformed_data_cache is None:
             raise ValueError("Chưa có dữ liệu cache. Hàm fit_transform() bị lỗi.")
             
        # Lấy thẳng dữ liệu từ RAM
        X = self._transformed_data_cache
        y = np.array(obj._labels)

        metrics_history = []
        
        # Dùng 'saga' solver: Thuật toán tối ưu cực tốt cho Dataset lớn
        solver_type = 'saga' if len(X) > 10000 else 'lbfgs'

        for i in range(n_repeats):
            # Cắt Train/Test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i*42)
            
            # Huấn luyện mô hình siêu tốc
            model = LogisticRegression(max_iter=500, solver=solver_type, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Thu thập chỉ số
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            metrics_history.append(metrics)

        avg_metrics = {k: np.mean([m[k] for m in metrics_history]) for k in metrics_history[0].keys()}
        print(f"[RESULT] {self._method} (Mất {time.time() - start_time:.1f}s) - F1: {avg_metrics['f1_score']:.4f}\n")
        return avg_metrics
    
    def visitImageDataset(self, obj: ImageDataset):
        self._dataset_name = "Image Dataset"
        try:
            self.fit_transform(obj)
            self._explanied_variance_sum = self._pca.explained_variance_.sum()
            self._explanied_variance_ratio_sum = self._pca.explained_variance_ratio_.sum()
            self._status = "Success"
        except Exception as e:
            self._status = f"Failed ({str(e)})"
        finally:
            self.log()
        return
    
    def run(self, obj: ImageDataset):
        if isinstance(obj, ImageDataset):
            self.visitImageDataset(obj)
        return
    
    def save_images(self, obj: ImageDataset, base_dir: str = "../data/preprocessing/color_space"):
        save_dir = os.path.join(base_dir, self._method.lower())
        print(f"   [INFO] Đang xuất file ảnh ra: {save_dir} ... (Vui lòng đợi)")
        
        code = COLOR_MAP[self._method]
        idx_to_class = {v: k for k, v in obj.class_idx.items()}
        
        for path, label, fname in zip(obj.image_paths, obj._labels, obj._file_names):
            class_name = idx_to_class[label]
            class_dir = os.path.join(save_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            img = cv2.imread(path)          # BGR
            if img is None:
                continue
                
            # FIX: convert BGR→RGB trước, rồi mới áp COLOR_MAP
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if code is not None:
                img = cv2.cvtColor(img, code)   # bây giờ đầu vào là RGB → đúng
            
            img = cv2.resize(img, (128, 128))
            
            # FIX: nếu là Grayscale thì không cần convert ngược
            # nếu là RGB/HSV/LAB thì convert RGB→BGR trước khi imwrite
            if self._method != "Grayscale":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(os.path.join(class_dir, fname), img)
            
        print(f"   [SUCCESS] Đã lưu xong ảnh cho không gian màu {self._method}!")