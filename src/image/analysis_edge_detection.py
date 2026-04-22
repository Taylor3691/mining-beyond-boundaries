import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

from core import Service
from image.dataset import ImageDataset
from config.settings import DEFAULT_SIZE

class EdgeDetectionAnalysis(Service):
    def __init__(self, n_samples: int = 1000):
        self._step_name = "Analysis: Edge Detection & Density"
        self._status = "Initialized"
        self.n_samples = n_samples
        self.results_df = None
        
    def _apply_edge_filters(self, img_gray):
        """Áp dụng 6 bộ lọc cạnh và trả về Mật độ cạnh (Edge Density)"""
        densities = {}
        total_pixels = img_gray.shape[0] * img_gray.shape[1]
        
        # 1. Canny (2 bộ tham số: Nhạy & Ít nhạy)
        edges_canny_1 = cv2.Canny(img_gray, 50, 150)
        edges_canny_2 = cv2.Canny(img_gray, 100, 200)
        densities['Canny (50, 150)'] = np.sum(edges_canny_1 > 0) / total_pixels
        densities['Canny (100, 200)'] = np.sum(edges_canny_2 > 0) / total_pixels
        
        # 2. Sobel (2 bộ tham số: Ngưỡng thấp & Ngưỡng cao)
        grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = cv2.magnitude(grad_x, grad_y)
        densities['Sobel (Threshold=50)'] = np.sum(sobel_mag > 50) / total_pixels
        densities['Sobel (Threshold=100)'] = np.sum(sobel_mag > 100) / total_pixels
        
        # 3. Prewitt (2 bộ tham số: Ngưỡng thấp & Ngưỡng cao)
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        prewitt_x = cv2.filter2D(img_gray, cv2.CV_64F, kernelx)
        prewitt_y = cv2.filter2D(img_gray, cv2.CV_64F, kernely)
        prewitt_mag = cv2.magnitude(prewitt_x, prewitt_y)
        densities['Prewitt (Threshold=50)'] = np.sum(prewitt_mag > 50) / total_pixels
        densities['Prewitt (Threshold=100)'] = np.sum(prewitt_mag > 100) / total_pixels
        
        return densities

    # HÀM BẮT BUỘC SỐ 1 THEO ABSTRACT CLASS
    def visitImageDataset(self, dataset: ImageDataset):
        print(f"[INFO] Bắt đầu trích xuất đặc trưng Cạnh trên mẫu {self.n_samples} ảnh...")
        
        all_paths = np.array(dataset.image_paths)
        all_labels = np.array(dataset._labels)
        
        if len(all_labels) > self.n_samples:
            _, sample_indices = train_test_split(
                np.arange(len(all_labels)), test_size=self.n_samples, 
                stratify=all_labels, random_state=42
            )
        else:
            sample_indices = np.arange(len(all_labels))
            
        class_names_dict = {v: k for k, v in dataset.class_idx.items()}
        records = []
        
        for count, idx in enumerate(sample_indices):
            path = all_paths[idx]
            label_idx = all_labels[idx]
            class_name = class_names_dict.get(label_idx, f"Class {label_idx}")
            
            img = cv2.imread(path)
            if img is None: continue
            
            img = cv2.resize(img, DEFAULT_SIZE)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Tính Edge Density cho 6 bộ lọc
            densities = self._apply_edge_filters(img_gray)
            
            for method, density in densities.items():
                records.append({
                    "Image_Index": idx,
                    "Class": class_name,
                    "Method": method,
                    "Edge_Density": density
                })
                
            if (count + 1) % 200 == 0:
                print(f"   -> Đã xử lý {count + 1}/{len(sample_indices)} ảnh")
                
        self.results_df = pd.DataFrame(records)
        self._status = "Success"
        print("[SUCCESS] Trích xuất Edge Density hoàn tất!")
        self.log()
        return self.results_df

    def run(self, dataset: ImageDataset):
        if isinstance(dataset, ImageDataset):
            return self.visitImageDataset(dataset)

    def log(self):
        print(f"[STATUS] {self._step_name} - {self._status}")