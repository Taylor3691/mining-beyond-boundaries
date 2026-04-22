import numpy as np
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # <--- Import thêm module này

from core import Visualization
from image.dataset import ImageDataset
from visualization.evaluation import plot_scree_pca, plot_pca_scatter_2d, plot_pca_scatter_3d
from config.settings import DEFAULT_SIZE

class FeatureSpaceAnalysis(Visualization):
    def __init__(self, n_samples: int = 2000):
        """
        Khởi tạo đối tượng và thiết lập giá trị ban đầu.
        
        Input:
            n_samples: Số lượng mẫu hoặc số lần lặp dùng trong xử lý/đánh giá.
        
        Output:
            None.
        """
        self._step_name = "Analysis: PCA & t-SNE Feature Space"
        self._dataset_name = ""
        self._status = "Initialized"
        self.n_samples = n_samples
        
    def visitImageDataset(self, obj: ImageDataset):
        """
        Thực thi xử lý dữ liệu theo kiểu đối tượng đầu vào.
        
        Input:
            obj: Đối tượng dữ liệu đầu vào cần được xử lý.
        
        Output:
            None.
        """
        self._dataset_name = "Image Dataset"
        print(f"[INFO] Bắt đầu nạp {self.n_samples} ảnh (lấy mẫu đều 10 class) vào RAM...")
        
        # 1. Trích xuất toàn bộ đường dẫn và nhãn từ Dataset
        all_paths = np.array(obj.image_paths)
        all_labels = np.array(obj._labels)
        
        # 2. Lấy mẫu ngẫu nhiên có phân tầng (Stratified Sampling)
        if len(all_labels) > self.n_samples:
            # Rút ra n_samples ảnh, đảm bảo tỷ lệ các class đều nhau
            _, sample_indices = train_test_split(
                np.arange(len(all_labels)), 
                test_size=self.n_samples, 
                stratify=all_labels, 
                random_state=42
            )
        else:
            sample_indices = np.arange(len(all_labels))
            
        images = []
        labels = []
        
        print("[INFO] Đang đọc và biến đổi ảnh...")
        for idx in sample_indices:
            path = all_paths[idx]
            lbl = all_labels[idx]
            
            img = cv2.imread(path)
            if img is not None:
                # Chuyển BGR (OpenCV) sang RGB để đúng chuẩn màu
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img, DEFAULT_SIZE)
                images.append(img_resized.flatten())
                labels.append(lbl)
                
        X = np.array(images)
        y = np.array(labels)
        
        # Xây dựng danh sách tên Class chuẩn xác từ obj.class_idx
        max_label = max(obj.class_idx.values())
        class_list = [f"Class {i}" for i in range(max_label + 1)]
        for name, idx in obj.class_idx.items():
            class_list[idx] = name
        
        print(f"[INFO] Dữ liệu thu thập: {X.shape[0]} ảnh. Tổng số chiều (Features): {X.shape[1]}")
        
        # 3. Chuẩn hóa dữ liệu (Standard Scaler)
        print("[INFO] Đang chuẩn hóa dữ liệu...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 4. Chạy PCA
        n_components = min(X.shape[0], X.shape[1])
        print(f"[INFO] Đang phân rã PCA để tìm {n_components} components...")
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # 5. Vẽ Scree Plot
        print("[INFO] Vẽ Scree Plot (Tỷ lệ phương sai)...")
        plot_scree_pca(pca.explained_variance_ratio_)
        
        # 6. Vẽ Scatter Plot 2D & 3D
        print("[INFO] Vẽ Scatter Plot 2D & 3D...")
        plot_pca_scatter_2d(X_pca[:, :2], y, class_list)
        if n_components >= 3:
            plot_pca_scatter_3d(X_pca[:, :3], y, class_list)
            
        # 7. Chiếu t-SNE
        print("[INFO] Đang chiếu t-SNE (Dựa trên Top 50 PCA components)...")
        X_for_tsne = X_pca[:, :50] if n_components > 50 else X_pca
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(X_for_tsne)
        
        # Vẽ t-SNE
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(y)
        palette = sns.color_palette("tab10", len(unique_labels))
        for i, lbl in enumerate(unique_labels):
            mask = y == lbl
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                        color=palette[i], label=class_list[lbl], alpha=0.7, edgecolors='w', s=50)
        plt.title("t-SNE Projection (Dữ liệu giảm nhiễu từ PCA)", fontsize=14, fontweight='bold')
        plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("tsne_scatter_2d.png", dpi=300)
        plt.show()
        plt.close()

        self._status = "Success"
        self.log()

    def run(self, obj: ImageDataset):
        """
        Thực thi quy trình xử lý chính của hàm.
        
        Input:
            obj: Đối tượng dữ liệu đầu vào cần được xử lý.
        
        Output:
            None.
        """
        if isinstance(obj, ImageDataset):
            self.visitImageDataset(obj)
            
    def log(self):
        """
        Ghi nhận và in thông tin trạng thái thực thi.
        
        Input:
            Không có.
        
        Output:
            None.
        """
        print(f"[STATUS] {self._step_name} - {self._status}")