from image import ImageDataset
from core import Preprocessing
from config import DEFAULT_N_COMPONENTS, SUPPORT_COLOR_SPACE, COLOR_MAP, DEFAULT_SIZE
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import time
import os


class ColorTransform(Preprocessing):
    def __init__(self, method: str = None, n: int = DEFAULT_N_COMPONENTS):
        """
        Khởi tạo lớp thực hiện biến đổi không gian màu và giảm chiều bằng PCA.
        
        Input:
            method: Phương pháp hoặc chế độ xử lý được sử dụng.
            n: Số thành phần chính giữ lại trong phép giảm chiều PCA.
        
        Output:
            None.
        """
        if method is None:
            raise ValueError("Cannot let the method empty")
        elif method not in SUPPORT_COLOR_SPACE:
            raise ValueError("This color space is not support")

        self._step_name = "Preprocessing: Color Space Transform"
        self._dataset_name = ""
        self._n = n
        self._scaler = StandardScaler()
        self._pca = IncrementalPCA(n_components=n)
        self._method = method
        return
    
    def fit(self, obj: ImageDataset):
        """
        Huấn luyện bộ chuẩn hóa (Scaler) và thuật toán PCA trên dữ liệu ảnh.
        
        Input:
            obj: Đối tượng dữ liệu đầu vào cần được xử lý.
        
        Output:
            None.
        """
        code = COLOR_MAP[self._method]
        for batch, _ in obj.load():
            images = []
            
            for img in batch:
                if code is not None:
                    img = cv2.cvtColor(img, code)
                img = cv2.resize(img, DEFAULT_SIZE)
                img = img.flatten()
                images.append(img)
        
            images = np.array(images)
            self._scaler.partial_fit(images)

        for batch, _ in obj.load():
            images = []

            for img in batch:
                if code is not None:
                    img = cv2.cvtColor(img, code)

                img = cv2.resize(img, DEFAULT_SIZE)
                img = img.flatten()

                images.append(img)

            images = np.array(images)

            images = self._scaler.transform(images)

            self._pca.partial_fit(images)
        return
    
    def transform(self, obj: ImageDataset):
        """
        Biến đổi không gian màu và áp dụng PCA để giảm chiều dữ liệu.
        
        Input:
            obj: Đối tượng dữ liệu đầu vào cần được xử lý.
        
        Output:
            Giá trị trả về của hàm.
        """
        code = COLOR_MAP[self._method]
        result = []
        for batch, _ in obj.load():
            images = []

            for img in batch:
                if code is not None:
                    img = cv2.cvtColor(img, code)

                img = cv2.resize(img, DEFAULT_SIZE)
                img = img.flatten()

                images.append(img)

            images = np.array(images)

            images = self._scaler.transform(images)
            images = self._pca.transform(images)
            result.append(images)

        return result
        
    def fit_transform(self, obj: ImageDataset):
        """
        Vừa huấn luyện vừa thực hiện biến đổi dữ liệu.
        
        Input:
            obj: Đối tượng dữ liệu đầu vào cần được xử lý.
        
        Output:
            Giá trị trả về của hàm.
        """
        self.fit(obj)
        return self.transform(obj)

    def log(self):
        """
        In thông tin tóm tắt về quá trình biến đổi màu và kết quả PCA.
        
        Input:
            Không có.
        
        Output:
            None.
        """
        print(f"Bước xử lý : {self._step_name}")
        print(f"Tập dữ liệu: {self._dataset_name}")
        print(f"Trạng thái : {self._status}")
        print(f"Không gian màu: {self._method}")
        print(f"Tổng phương sai giải thích theo PCA với k={self._n}: {self._explanied_variance_sum}")
        print(f"Tổng tỷ lệ phương sai giải thích:{self._explanied_variance_ratio_sum}")
        for i, var in enumerate(self._pca.explained_variance_ratio_[:5]):
            print(f"PC{i+1}: {var:.6f}")
    
    def visitImageDataset(self, obj: ImageDataset):
        """
        Thực thi quy trình biến đổi không gian màu trên ImageDataset.
        
        Input:
            obj: Đối tượng dữ liệu đầu vào cần được xử lý.
        
        Output:
            None.
        """
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
        """
        Hàm chạy chính của quy trình tiền xử lý màu.
        
        Input:
            obj: Đối tượng dữ liệu đầu vào cần được xử lý.
        
        Output:
            None.
        """
        if isinstance(obj, ImageDataset):
            self.visitImageDataset(obj)
        return 

class ColorTransformEvaluator(ColorTransform):
    """
    Class kế thừa từ ColorTransform gốc.
    Mở rộng thêm tính năng: Ép RAM siêu tốc, Huấn luyện Logistic Regression và Xuất file ảnh.
    """
    def __init__(self, method: str = None, n: int = DEFAULT_N_COMPONENTS):
        """
        Khởi tạo bộ đánh giá biến đổi màu, tối ưu cho việc cache dữ liệu vào RAM.
        
        Input:
            method: Phương pháp hoặc chế độ xử lý được sử dụng.
            n: Số thành phần chính giữ lại trong phép giảm chiều PCA.
        
        Output:
            None.
        """
        # Kế thừa hoàn toàn hàm __init__ của class cha
        super().__init__(method=method, n=n)
        # Tăng batch_size để tối ưu tốc độ
        self._pca = IncrementalPCA(n_components=n, batch_size=1024)
        self._transformed_data_cache = None 

    def visitImageDataset(self, obj: ImageDataset):
        """
        Ghi đè hàm xử lý dataset để lưu trữ dữ liệu đã biến đổi vào cache.
        
        Input:
            obj: Đối tượng dữ liệu đầu vào cần được xử lý.
        
        Output:
            None.
        """
        self._dataset_name = "Image Dataset"
        print(f"   [INFO] Đang chạy Tiền xử lý (Fast-Cache Mode)...")
        try:
            self.fit(obj)
            batches = self.transform(obj)
            self._transformed_data_cache = np.vstack(batches)
            
            self._explanied_variance_sum = self._pca.explained_variance_.sum()
            self._explanied_variance_ratio_sum = self._pca.explained_variance_ratio_.sum()
            self._status = "Success"
        except Exception as e:
            self._status = f"Failed ({str(e)})"
        finally:
            self.log()
        return
            
    def evaluation(self, obj: ImageDataset, n_repeats: int = 3):
        """
        Thực hiện đánh giá hiệu quả của không gian màu thông qua mô hình Logistic Regression.
        
        Input:
            obj: Đối tượng dữ liệu đầu vào cần được xử lý.
            n_repeats: Số lượng mẫu hoặc số lần lặp dùng trong xử lý/đánh giá.
        
        Output:
            Giá trị trả về của hàm.
        """
        print(f"\n[EVALUATION] Bắt đầu huấn luyện mô hình (Không gian: {self._method})...")
        start_time = time.time()
        
        if self._transformed_data_cache is None:
             raise ValueError("Chưa có dữ liệu cache. Hàm run() bị lỗi.")
             
        X = self._transformed_data_cache
        y = np.array(obj._labels)
        metrics_history = []
        solver_type = 'saga' if len(X) > 10000 else 'lbfgs'

        for i in range(n_repeats):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i*42)
            model = LogisticRegression(max_iter=500, solver=solver_type, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
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
    
    def save_images(self, obj: ImageDataset, base_dir: str = "../data/preprocessing/color_space"):
        """
        Lưu các mẫu ảnh đã được biến đổi không gian màu ra ổ cứng.
        
        Input:
            obj: Đối tượng dữ liệu đầu vào cần được xử lý.
            base_dir: Đường dẫn tệp hoặc thư mục liên quan đến dữ liệu.
        
        Output:
            None.
        """
        save_dir = os.path.join(base_dir, self._method.lower())
        print(f"   [INFO] Đang xuất file ảnh ra: {save_dir} ... (Vui lòng đợi)")
        
        code = COLOR_MAP[self._method]
        idx_to_class = {v: k for k, v in obj.class_idx.items()}
        
        for path, label, fname in zip(obj.image_paths, obj._labels, obj._file_names):
            class_name = idx_to_class[label]
            class_dir = os.path.join(save_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            img = cv2.imread(path)
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if code is not None:
                img = cv2.cvtColor(img, code) 
            
            img = cv2.resize(img, (128, 128))
            
            if self._method != "Grayscale":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(os.path.join(class_dir, fname), img)
            
        print(f"   [SUCCESS] Đã lưu xong ảnh cho không gian màu {self._method}!")