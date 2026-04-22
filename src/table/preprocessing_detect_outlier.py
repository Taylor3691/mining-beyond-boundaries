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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler 
from statistic.test_distribution import KolmogorovSmirnovTesting
from utils.file import jaccard_similarity



class BaseOutlierDetector(Preprocessing):
    """
    Class trung gian xử lý boilerplate code cho việc phát hiện và loại bỏ ngoại lai.
    """
    def __init__(self, step_name: str):
        """
        Khởi tạo BaseOutlierDetector với cấu hình mặc định.

        Input:
            step_name: Tên bước xử lý (để log và nhận diện).

        Output:
            None.
        """
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
        """
        Kiểm tra và lưu số lượng mẫu ban đầu.

        Input:
            df: DataFrame chứa dữ liệu cần phân tích.

        Output:
            None.
        """
        if df is None or df.empty:
            raise ValueError("Dữ liệu trống, không thể fit.")
        self._metadata["original_count"] = len(df)

    def _get_outlier_mask(self, numeric_df: pd.DataFrame) -> pd.Series:
        """
        Tìm các dòng ngoại lai (abstract method - lớp con phải implement).

        Input:
            numeric_df: DataFrame chỉ chứa các cột số.

        Output:
            pd.Series (boolean): True nếu dòng là ngoại lai.
        """
        raise NotImplementedError("Phải implement hàm _get_outlier_mask ở class con")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tìm, lưu index và loại bỏ các dòng ngoại lai khỏi DataFrame.

        Input:
            df: DataFrame gốc cần loại ngoại lai.

        Output:
            pd.DataFrame: Dữ liệu đã loại bỏ các dòng ngoại lai.
        """
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
        """
        Kết hợp fit và transform trong một bước.

        Input:
            df: DataFrame gốc cần xử lý.

        Output:
            pd.DataFrame: Dữ liệu đã loại bỏ ngoại lai.
        """
        self.fit(df)
        return self.transform(df)
    
    def visitImageDataset(self, obj):
        """
        Không hỗ trợ dữ liệu hình ảnh (bắt buộc do kế thừa từ Service).

        Input:
            obj: Đối tượng ImageDataset.

        Output:
            None.
        """
        pass

    def visitTableDataset(self, obj):
        """
        Triển khai pipeline phát hiện ngoại lai lên đối tượng TableDataset.

        Input:
            obj: Đối tượng TableDataset chứa dữ liệu cần xử lý.

        Output:
            None (cập nhật trực tiếp dữ liệu trong obj).
        """
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
        """
        Điểm vào thực thi - gọi visitTableDataset.

        Input:
            obj: Đối tượng TableDataset.

        Output:
            None.
        """
        self.visitTableDataset(obj)

    def log(self):
        """
        In thông tin tổng quan và tỉ lệ phát hiện ngoại lai.

        Input:
            Không có.

        Output:
            None (in ra màn hình).
        """
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
        """
        Phát hiện ngoại lai bằng phương pháp IQR.

        Input:
            numeric_df: DataFrame chỉ chứa các cột số.

        Output:
            pd.Series (boolean): True nếu dòng vi phạm ngưỡng IQR.
        """
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        
        # Nếu bất kỳ cột số nào của dòng đó vi phạm ngưỡng IQR thì bị đánh dấu là ngoại lai
        mask = ((numeric_df < (Q1 - self.multiplier * IQR)) | (numeric_df > (Q3 + self.multiplier * IQR))).any(axis=1)
        return mask

# Z-Score
class ZScore_Outlier(BaseOutlierDetector):
    """Phát hiện ngoại lai bằng phương pháp Z-Score."""

    def __init__(self, threshold: float = 3.0):
        """
        Khởi tạo bộ phát hiện ngoại lai Z-Score.

        Input:
            threshold: Ngưỡng z-score (mặc định 3.0).

        Output:
            None.
        """
        super().__init__(step_name=f"Z-Score Outlier Detection (threshold={threshold})")
        self.threshold = threshold

    def _get_outlier_mask(self, numeric_df: pd.DataFrame) -> pd.Series:
        """
        Phát hiện ngoại lai bằng z-score.

        Input:
            numeric_df: DataFrame chỉ chứa các cột số.

        Output:
            pd.Series (boolean): True nếu dòng có z-score vượt ngưỡng.
        """
        # Tính z-score, bỏ qua NaN
        z_scores = np.abs(stats.zscore(numeric_df, nan_policy='omit'))
        
        # Những dòng có z-score > threshold ở bất kỳ cột nào sẽ là ngoại lai
        mask = (z_scores > self.threshold).any(axis=1)
        return mask

# Isolation Forest
class IForest_Outlier(BaseOutlierDetector):
    """Phát hiện ngoại lai bằng Isolation Forest."""

    def __init__(self, contamination: float = 0.05):
        """
        Khởi tạo bộ phát hiện ngoại lai Isolation Forest.

        Input:
            contamination: Tỉ lệ ngoại lai ước tính (0.01, 0.05, 0.1).

        Output:
            None.
        """
        if contamination not in [0.01, 0.05, 0.1]:
            print(f"Cảnh báo: Contamination {contamination} không nằm trong yêu cầu {{0.01, 0.05, 0.1}}")
            
        super().__init__(step_name=f"Isolation Forest (contamination={contamination})")
        self.contamination = contamination

    def _get_outlier_mask(self, numeric_df: pd.DataFrame) -> pd.Series:
        """
        Phát hiện ngoại lai bằng mô hình Isolation Forest.

        Input:
            numeric_df: DataFrame chỉ chứa các cột số.

        Output:
            pd.Series (boolean): True nếu dòng được đánh giá là ngoại lai (-1).
        """
        model = IsolationForest(contamination=self.contamination, random_state=42)
        preds = model.fit_predict(numeric_df)
        
        # Kết quả: -1 là ngoại lai, 1 là bình thường
        return pd.Series(preds == -1, index=numeric_df.index)

# Local Outlier Factor (LOF)
class LOF_Outlier(BaseOutlierDetector):
    """Phát hiện ngoại lai bằng Local Outlier Factor (LOF)."""

    def __init__(self, n_neighbors: int = 20):
        """
        Khởi tạo bộ phát hiện ngoại lai LOF.

        Input:
            n_neighbors: Số láng giềng (10, 20, 50).

        Output:
            None.
        """
        if n_neighbors not in [10, 20, 50]:
            print(f"Cảnh báo: n_neighbors {n_neighbors} không nằm trong yêu cầu {{10, 20, 50}}")
            
        super().__init__(step_name=f"Local Outlier Factor (n_neighbors={n_neighbors})")
        self.n_neighbors = n_neighbors

    def _get_outlier_mask(self, numeric_df: pd.DataFrame) -> pd.Series:
        """
        Phát hiện ngoại lai bằng mô hình LOF.

        Input:
            numeric_df: DataFrame chỉ chứa các cột số.

        Output:
            pd.Series (boolean): True nếu dòng được đánh giá là ngoại lai (-1).
        """
        model = LocalOutlierFactor(n_neighbors=self.n_neighbors)
        preds = model.fit_predict(numeric_df)
        
        # Kết quả: -1 là ngoại lai, 1 là bình thường
        return pd.Series(preds == -1, index=numeric_df.index)
    
# DBSCAN
class DBSCAN_Outlier(BaseOutlierDetector):
    """Phát hiện ngoại lai bằng DBSCAN với sub-sampling."""

    def __init__(self, eps: float = 1.5, min_samples: int = 10, sample_size: int = 40000):
        """
        Khởi tạo bộ phát hiện ngoại lai DBSCAN.

        Input:
            eps: Bán kính láng giềng DBSCAN.
            min_samples: Số mẫu tối thiểu để tạo core point.
            sample_size: Kích thước mẫu con để tối ưu bộ nhớ.

        Output:
            None.
        """
        super().__init__(step_name=f"DBSCAN (eps={eps}, min_samples={min_samples}, Sub-sample={sample_size})")
        self.eps = eps
        self.min_samples = min_samples
        self.sample_size = sample_size

    def _get_outlier_mask(self, numeric_df: pd.DataFrame) -> pd.Series:
        """
        Phát hiện ngoại lai bằng DBSCAN với sub-sampling và Nearest Neighbors.

        Input:
            numeric_df: DataFrame chỉ chứa các cột số.

        Output:
            pd.Series (boolean): True nếu dòng nằm ngoài các cụm DBSCAN.
        """
        # Chuẩn hoá dữ liệu trước tiên
        numeric_df_32 = numeric_df.astype(np.float32)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df_32)
        
        del numeric_df_32
        gc.collect()

        total_rows = scaled_data.shape[0]
        actual_sample_size = min(self.sample_size, total_rows)

        # Lấy mẫu đại diện
        np.random.seed(42) # Cố định seed để kết quả tái lập được
        sample_indices = np.random.choice(total_rows, size=actual_sample_size, replace=False)
        sampled_data = scaled_data[sample_indices]

        # Chạy DBSCAN trên tập mẫu 
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=1)
        dbscan.fit(sampled_data)

        # Trích xuất điểm lõi
        # dbscan.components_ chứa tọa độ các điểm đủ điều kiện làm Lõi của các cụm
        core_samples = dbscan.components_
        
        del sampled_data
        gc.collect()

        # Kiểm tra nếu không có cụm nào được tạo ra
        if len(core_samples) == 0:
            print("  CẢNH BÁO: DBSCAN không tìm thấy cụm nào! Tất cả đều bị coi là ngoại lai. Hãy tăng eps lên.")
            return pd.Series(True, index=numeric_df.index)

        # Dùng nearest neighbors 
        # Tìm khoảng cách tới 1 điểm lõi gần nhất
        nn = NearestNeighbors(n_neighbors=1, n_jobs=1)
        nn.fit(core_samples)
        
        distances, _ = nn.kneighbors(scaled_data)
        
        # Nếu khoảng cách đến Lõi gần nhất > eps => Đây là điểm nhiễu (Ngoại lai)
        outlier_mask = (distances.flatten() > self.eps)

        # Dọn rác lần cuối
        del scaled_data, distances, core_samples
        gc.collect()

        return pd.Series(outlier_mask, index=numeric_df.index)
    
def main():
    """
    Hàm chính test pipeline phát hiện ngoại lai với nhiều phương pháp và đánh giá Jaccard.

    Input:
        Không có.

    Output:
        None (in kết quả ra màn hình).
    """
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