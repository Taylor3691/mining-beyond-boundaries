import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

import sys
import os
import gc

from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import PATH_FOLDER_TABLE_RAW
from core.service_base import Preprocessing
from table.dataset import TableDataset
from config import settings 

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from statistic.test_distribution import KolmogorovSmirnovTesting
from utils.file import jaccard_similarity



class BaseOutlierDetector(Preprocessing):
    """
    Class trung gian xử lý boilerplate code cho việc phát hiện và loại bỏ ngoại lai.
    """
    def __init__(self, step_name: str):
        self._step_name = step_name
        self._dataset_path = "Unknown"
        self._status = "Pending"
        self._error_message = ""
        self._outlier_indices = []  # Lưu lại index để tính Jaccard similarity sau này
        
        self._metadata = {
            "original_count": 0,
            "outlier_count": 0,
            "detection_rate": 0.0
        }

    @property
    def outlier_indices(self):
        """Property giữ lại thông tin vị trí các phần tử được cho là ngoại lai"""
        return self._outlier_indices

    def fit(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise ValueError("Dữ liệu trống, không thể fit.")
        self._metadata["original_count"] = len(df)

    def _get_outlier_mask(self, numeric_df: pd.DataFrame) -> pd.Series:
        """Hàm abstract ảo để các class con implement logic tìm ngoại lai"""
        raise NotImplementedError("Phải implement hàm _get_outlier_mask ở class con")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hàm tìm, lưu index và loại bỏ các phần tử ngoại lai"""
        # Chỉ áp dụng phát hiện ngoại lai trên các cột dữ liệu số
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Xử lý NaN tạm thời để các model sklearn không bị crash khi predict
        numeric_df_filled = numeric_df.fillna(numeric_df.mean())
        
        # Lấy boolean mask từ thuật toán (True là ngoại lai)
        outlier_mask = self._get_outlier_mask(numeric_df_filled)
        
        # Cập nhật danh sách index và thống kê
        self._outlier_indices = df[outlier_mask].index.tolist()
        self._metadata["outlier_count"] = len(self._outlier_indices)
        
        if self._metadata["original_count"] > 0:
            self._metadata["detection_rate"] = (self._metadata["outlier_count"] / self._metadata["original_count"]) * 100
            
        # Trả về dataframe đã drop các dòng ngoại lai
        return df.drop(index=self._outlier_indices)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
    
    def visitImageDataset(self, obj):
        """Hàm này bắt buộc phải có do kế thừa từ Service, nhưng không dùng cho Table"""
        pass

    def visitTableDataset(self, obj):
        """Triển khai pipeline lên TableDataset"""
        self._dataset_path = getattr(obj, '_folder_path', "Unknown Path")
        
        try:
            # Lưu lại tên cột target để set lại sau khi transform
            target_col = getattr(obj, '_target_column', None)
            
            # Chạy pipeline xử lý
            clean_df = self.fit_transform(obj.data)
            
            # Cập nhật lại dữ liệu sạch vào object
            obj.data = clean_df 
            if target_col:
                obj.set_target(target_col)
                
            self._status = "Success"
        except Exception as e:
            self._status = "Failed"
            self._error_message = str(e)

    def run(self, obj):
        self.visitTableDataset(obj)

    def log(self):
        """In ra thông tin tổng quan và tỉ lệ phát hiện"""
        print("\n" + "="*50)
        print(f"1. Processing Step : {self._step_name}")
        print(f"2. Target Dataset  : {self._dataset_path}")
        print(f"3. Status          : {self._status}")
        
        if self._status == "Success":
            print(f"4. Result Output:")
            print(f"   - Original shape    : {self._metadata.get('original_count', 0)} rows")
            print(f"   - Outliers detected : {self._metadata.get('outlier_count', 0)} rows")
            print(f"   - Detection Rate    : {self._metadata.get('detection_rate', 0.0):.4f}%")
        else:
            print(f"4. Error Details   : {self._error_message}")
        print("="*50 + "\n")


# IQR (Interquartile Range)
class IQR_Outlier(BaseOutlierDetector):
    def __init__(self, multiplier: float = 1.5):
        super().__init__(step_name=f"IQR Outlier Detection (multiplier={multiplier})")
        self.multiplier = multiplier

    def _get_outlier_mask(self, numeric_df: pd.DataFrame) -> pd.Series:
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        
        # Nếu bất kỳ cột số nào của dòng đó vi phạm ngưỡng IQR thì bị đánh dấu là ngoại lai
        mask = ((numeric_df < (Q1 - self.multiplier * IQR)) | (numeric_df > (Q3 + self.multiplier * IQR))).any(axis=1)
        return mask

# Z-Score
class ZScore_Outlier(BaseOutlierDetector):
    def __init__(self, threshold: float = 3.0):
        super().__init__(step_name=f"Z-Score Outlier Detection (threshold={threshold})")
        self.threshold = threshold

    def _get_outlier_mask(self, numeric_df: pd.DataFrame) -> pd.Series:
        # Tính z-score, bỏ qua NaN
        z_scores = np.abs(stats.zscore(numeric_df, nan_policy='omit'))
        
        # Những dòng có z-score > threshold ở bất kỳ cột nào sẽ là ngoại lai
        mask = (z_scores > self.threshold).any(axis=1)
        return mask

# Isolation Forest
class IForest_Outlier(BaseOutlierDetector):
    def __init__(self, contamination: float = 0.05):
        if contamination not in [0.01, 0.05, 0.1]:
            print(f"Cảnh báo: Contamination {contamination} không nằm trong yêu cầu {{0.01, 0.05, 0.1}}")
            
        super().__init__(step_name=f"Isolation Forest (contamination={contamination})")
        self.contamination = contamination

    def _get_outlier_mask(self, numeric_df: pd.DataFrame) -> pd.Series:
        model = IsolationForest(contamination=self.contamination, random_state=42)
        preds = model.fit_predict(numeric_df)
        
        # Kết quả: -1 là ngoại lai, 1 là bình thường
        return pd.Series(preds == -1, index=numeric_df.index)

# Local Outlier Factor (LOF)
class LOF_Outlier(BaseOutlierDetector):
    def __init__(self, n_neighbors: int = 20):
        if n_neighbors not in [10, 20, 50]:
            print(f"Cảnh báo: n_neighbors {n_neighbors} không nằm trong yêu cầu {{10, 20, 50}}")
            
        super().__init__(step_name=f"Local Outlier Factor (n_neighbors={n_neighbors})")
        self.n_neighbors = n_neighbors

    def _get_outlier_mask(self, numeric_df: pd.DataFrame) -> pd.Series:
        model = LocalOutlierFactor(n_neighbors=self.n_neighbors)
        preds = model.fit_predict(numeric_df)
        
        # Kết quả: -1 là ngoại lai, 1 là bình thường
        return pd.Series(preds == -1, index=numeric_df.index)

# DBSCAN
class DBSCAN_Outlier(BaseOutlierDetector):
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        super().__init__(step_name=f"DBSCAN Outlier Detection (eps={eps}, min_samples={min_samples})")
        self.eps = eps
        self.min_samples = min_samples

    def _get_outlier_mask(self, numeric_df: pd.DataFrame) -> pd.Series:
        model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        preds = model.fit_predict(numeric_df)
        
        # Các điểm nhiễu (ngoại lai) sẽ được gán nhãn là -1
        return pd.Series(preds == -1, index=numeric_df.index)
    
# Ham main dung de test
def main():
    CSV_PATH = str(Path(settings.PATH_FOLDER_TABLE_RAW) / "Building_Permits.csv")
    
    print("="*70)
    print(f" BƯỚC 1: KHỞI TẠO DỮ LIỆU TỪ FILE:\n {CSV_PATH}")
    print("="*70)

    # 1. Validate đường dẫn
    if not os.path.exists(CSV_PATH):
        print(f"LỖI: Không tìm thấy file tại '{CSV_PATH}'")
        print("Vui lòng kiểm tra lại xem file Building_Permits.csv đã nằm đúng trong thư mục data/table/ chưa.")
        return

    # 2. Khởi tạo Dataset
    dataset = TableDataset(path=CSV_PATH) 
    
    # 3. Chặn lỗi NoneType
    if dataset.data is None:
        print("LỖI: Không thể nạp dữ liệu. Hãy kiểm tra hàm load_table trong utils/file.py")
        return

    # 4. Gán _origin_data cho K-S Test
    dataset._origin_data = dataset.data.copy() 
    dataset.info()

    # BƯỚC 2: CÀI ĐẶT CÁC PHƯƠNG PHÁP 
    detectors = {
        "IQR": IQR_Outlier(multiplier=1.5),
        "Z-Score": ZScore_Outlier(threshold=3.0),
        "IForest (c=0.01)": IForest_Outlier(contamination=0.01),
        "IForest (c=0.05)": IForest_Outlier(contamination=0.05),
        "IForest (c=0.10)": IForest_Outlier(contamination=0.1),
        "LOF (n=10)": LOF_Outlier(n_neighbors=10),
        "LOF (n=20)": LOF_Outlier(n_neighbors=20),
        "LOF (n=50)": LOF_Outlier(n_neighbors=50),
        "DBSCAN": DBSCAN_Outlier(eps=4.0, min_samples=10) 
    }

    outlier_sets = {}

    print("\n" + "="*70)
    print(" BƯỚC 3: THỰC THI PHÁT HIỆN NGOẠI LAI & KIỂM ĐỊNH K-S TEST")
    print("="*70)
    
    for name, detector in detectors.items():
        print(f"\n>>> ĐANG CHẠY PHƯƠNG PHÁP: {name} <<<")
        
        # Clone dataset để test độc lập
        current_dataset = dataset.clone()
        current_dataset._origin_data = dataset._origin_data.copy()
        
        # Chạy thuật toán 
        detector.run(current_dataset)
        detector.log()
        
        # Lưu index ngoại lai
        outlier_sets[name] = detector.outlier_indices
        
        # Kiểm định KS
        ks_tester = KolmogorovSmirnovTesting(alpha=0.05)
        ks_tester.run(current_dataset)
        ks_tester.log()


    print("\n" + "="*70)
    print(" BƯỚC 4: ĐÁNH GIÁ ĐỘ CHỒNG CHÉO (JACCARD SIMILARITY)")
    print("="*70)
    
    pairs = list(combinations(detectors.keys(), 2))
    jaccard_results = []
    
    for method1, method2 in pairs:
        sim = jaccard_similarity(outlier_sets[method1], outlier_sets[method2])
        jaccard_results.append((method1, method2, sim))
    
    jaccard_results.sort(key=lambda x: x[2], reverse=True)
    
    print(f"{'Phương pháp 1':<20} | {'Phương pháp 2':<20} | {'Tỉ lệ (Jaccard)'}")
    print("-" * 65)
    for m1, m2, sim in jaccard_results:
        if sim > 0:
            print(f"{m1:<20} | {m2:<20} | {sim*100:>5.2f}%")

if __name__ == "__main__":
    main()