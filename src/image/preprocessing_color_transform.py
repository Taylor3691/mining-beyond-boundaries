from image import ImageDataset
from core import Preprocessing
from config import DEFAULT_N_COMPONENTS, SUPPORT_COLOR_SPACE, COLOR_MAP, DEFAULT_SIZE
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import cv2

class ColorTransform(Preprocessing):
    def __init__(self, method: str = None, n: int = DEFAULT_N_COMPONENTS):
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
        self.fit(obj)
        return self.transform(obj)

    def log(self):
        print(f"Bước xử lý : {self._step_name}")
        print(f"Tập dữ liệu: {self._dataset_name}")
        print(f"Trạng thái : {self._status}")
        print(f"Không gian màu: {self._method}")
        print(f"Tổng phương sai giải thích theo PCA với k={self._n}: {self._explanied_variance_sum}")
        print(f"Tổng tỷ lệ phương sai giải thích:{self._explanied_variance_ratio_sum}")
        for i, var in enumerate(self._pca.explained_variance_ratio_[:5]):
            print(f"PC{i+1}: {var:.6f}")
    
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
