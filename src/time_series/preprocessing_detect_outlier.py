import numpy as np
import pandas as pd
from typing import Any, Dict, List
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL
from core.service_base import Preprocessing
import os
from dataset import TimeSeriesDataset

class DetectOutlierTimeSeries(Preprocessing):
    def __init__(self,
                 method: str = 'z-score',
                 window_size: int = 7,  # Chu kỳ 7 ngày cho Covid-19
                 threshold: float = 3.0, # Ngưỡng Z-score
                 contamination: float = 0.05 # Tỷ lệ dị thường cho Isolation Forest
                 ):
        
        self._supported_methods = ['z-score', 'iforest', 'stl']
        if method not in self._supported_methods:
            raise ValueError(f"Method '{method}' not supported. Choose from {self._supported_methods}")
        
        self._method = method
        self._window_size = window_size
        self._threshold = threshold
        self._contamination = contamination
        self._stats: Dict[str, Any] = {}
        self._anomalies_mask = None # Lưu mảng boolean [True, False, ...]
        self._anomaly_indices: List[int] = [] # Lưu danh sách các index là dị thường
        return

    @property
    def method(self):
        return self._method

    def set_method(self, new_method: str):
        if new_method not in self._supported_methods:
            raise ValueError(f"Method '{new_method}' not supported.")
        self._method = new_method
        self._stats.clear()
        self._anomalies_mask = None
        self._anomaly_indices = []
        print(f"Đã chuyển phương pháp phát hiện dị thường sang: {self._method}")

    def get_anomaly_candidates(self) -> List[int]:
        if self._anomaly_indices is None or len(self._anomaly_indices) == 0:
            print("Chưa có điểm dị thường nào được phát hiện. Vui lòng gọi fit() hoặc run() trước.")
        return self._anomaly_indices

    def get_anomaly_mask(self) -> np.ndarray:
        return self._anomalies_mask

    def fit(self, series: pd.Series):
        """Tính toán và tìm các điểm khả nghi dựa trên phương pháp đã chọn"""
        series_clean = series.bfill().ffill()

        if self._method == 'z-score':
            self._fit_zscore_deseasonalized(series_clean)
        elif self._method == 'iforest':
            self._fit_isolation_forest(series_clean)
        elif self._method == 'stl':
            self._fit_stl_thresholding(series_clean)
        self._anomaly_indices = np.where(self._anomalies_mask)[0].tolist()

    def transform(self, arr: np.ndarray):
        # Với bài toán Anomaly Detection, transform có thể trả về mảng đã được fill (thay thế dị thường)
        # Trong giới hạn yêu cầu, ta chỉ tập trung detect, nên transform trả về mask dị thường
        pass

    def fit_transform(self, arr: np.ndarray):
        pass

    """
    def run(self, obj):
        Chạy thông qua Pipeline (nhận TimeSeriesDataset)S
        # Tránh import vòng, kiểm tra thuộc tính target
        if not hasattr(obj, 'target') or obj.target is None:
            raise ValueError("Dataset chưa được set_target().")
        
        try:
            target_series = obj.target
            self.fit(target_series)
            obj._anomaly_mask = self._anomalies_mask # Gắn mask vào dataset nếu cần
        except Exception as e:
            print(f"Error during outlier detection: {e}")
        finally:
            self.log()
        return
    """
    def run(self, obj):
        """Định tuyến (Route) đúng loại dataset vào đúng hàm visit"""
        # Nếu đang dùng base TimeSeriesDataset
        if obj.__class__.__name__ == "TimeSeriesDataset":
            self.visitTimeSeriesDataset(obj)
        else:
            print(f"[CẢNH BÁO] DetectOutlierTimeSeries không hỗ trợ loại dữ liệu: {type(obj)}")
        return

    def visitTimeSeriesDataset(self, obj):
        """Hàm xử lý chính cho TimeSeries Dataset"""
        if getattr(obj, 'target', None) is None:
            raise ValueError("Dataset chưa được gọi hàm set_target().")
        
        try:
            target_series = obj.target
            self.fit(target_series)
            obj._anomaly_mask = self._anomalies_mask # Gắn mask ngược lại dataset (nếu cần)
        except Exception as e:
            print(f"Error during outlier detection: {e}")
        finally:
            self.log()
        return

    def visitImageDataset(self, obj):
        """Không hỗ trợ dữ liệu hình ảnh."""
        pass

    def log(self):
        """In thông tin tóm tắt kết quả phát hiện dị thường."""
        print(f"--- DetectOutlierTimeSeries Log ---")
        print(f"Method: {self._method}")
        print(f"Total anomalies detected: {len(self._anomaly_indices)}")
        if self._method == 'z-score':
            print(f"  - Mean (Deseasonalized): {self._stats.get('mean', 0):.4f}")
            print(f"  - Std (Deseasonalized): {self._stats.get('std', 0):.4f}")
        elif self._method == 'stl':
            print(f"  - IQR Range: [{self._stats.get('lower_bound', 0):.4f}, {self._stats.get('upper_bound', 0):.4f}]")
        return

    def _fit_zscore_deseasonalized(self, series: pd.Series):
        # 1. Phân rã để lấy seasonality
        stl = STL(series, period=self._window_size, robust=True)
        res = stl.fit()
        
        # 2. Khử tính mùa vụ
        deseasonalized = series - res.seasonal
        
        # 3. Tính Z-score trên phần đã khử
        mean_val = deseasonalized.mean()
        std_val = deseasonalized.std()
        
        z_scores = (deseasonalized - mean_val) / (std_val + 1e-8)
        self._anomalies_mask = np.abs(z_scores) > self._threshold
        
        self._stats['mean'] = mean_val
        self._stats['std'] = std_val

    def _fit_isolation_forest(self, series: pd.Series):
        # Trích xuất đặc trưng trên cửa sổ trượt (Sliding Window)
        df_roll = pd.DataFrame({'y': series})
        for i in range(1, self._window_size):
            df_roll[f'y_lag_{i}'] = df_roll['y'].shift(i)
        
        df_roll = df_roll.fillna(method='bfill')
        X = df_roll.values

        # Khởi tạo và fit iForest
        model = IsolationForest(contamination=self._contamination, random_state=42)
        preds = model.fit_predict(X)
        
        # Isolation Forest trả về -1 cho dị thường, 1 cho bình thường
        self._anomalies_mask = (preds == -1)

    def _fit_stl_thresholding(self, series: pd.Series):
        # 1. Phân rã STL
        stl = STL(series, period=self._window_size, robust=True)
        res = stl.fit()
        resid = res.resid
        
        # 2. Xác định ngưỡng Thresholding trên phần Residual dựa vào IQR
        q1 = np.nanpercentile(resid, 25)
        q3 = np.nanpercentile(resid, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        self._anomalies_mask = (resid < lower_bound) | (resid > upper_bound)
        self._stats['lower_bound'] = lower_bound
        self._stats['upper_bound'] = upper_bound



from visualization.comparison import plot_anomalies_single_method, plot_anomalies_all_methods

def main():
    print("="*50)
    print(" BẮT ĐẦU PIPELINE PHÁT HIỆN DỊ THƯỜNG COVID-19")
    print("="*50)

    # 1. Khởi tạo và nạp dữ liệu
    # Lưu ý: Sửa 'path/to/covid_19_data.csv' thành đường dẫn thực tế của bạn
    data_path = './data/time-series/time-series-19-covid-combined.csv' 
    time_col = 'Date' # Cột thời gian thường thấy trong dataset Covid-19 Kaggle
    target_col = 'Confirmed'     # Cột cần phân tích (Có thể là 'New_cases' nếu bạn đã tính sai phân)
    
    # Tạo mock data nếu file không tồn tại để code không bị lỗi khi test nhanh
    if not os.path.exists(data_path):
        print(f"[CẢNH BÁO] Không tìm thấy {data_path}. Đang tạo dữ liệu mẫu...")
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        import numpy as np
        # Tạo trend + seasonality + noise + 2 điểm dị thường lớn
        values = np.linspace(10, 100, 100) + np.sin(np.arange(100)) * 10 + np.random.normal(0, 2, 100)
        values[30] += 50  # Dị thường 1
        values[75] -= 40  # Dị thường 2
        pd.DataFrame({time_col: dates, target_col: values}).to_csv(data_path, index=False)

    try:
        dataset = TimeSeriesDataset(path=data_path, time_column=time_col)
        dataset.info()
        
        # Set Target (Chọn biến cần phân tích dị thường)
        dataset.set_target(target_column=target_col)
        print(f"Đã set target: '{target_col}'\n")

    except Exception as e:
        print(f"[LỖI] Khởi tạo dataset thất bại: {e}")
        return

    # 2. Khởi tạo Service Phát hiện dị thường
    # Sử dụng window_size = 7 để bắt chu kỳ tuần của Covid-19
    detector = DetectOutlierTimeSeries(window_size=7, threshold=3.0, contamination=0.05)
    
    # Dictionary để lưu mask của tất cả các phương pháp phục vụ cho việc so sánh ở cuối
    all_masks_dict = {}
        
    # Danh sách 3 phương pháp cần test
    methods_to_test = ['z-score', 'iforest', 'stl']

    # 3. Vòng lặp chạy và đánh giá từng phương pháp
    for method in methods_to_test:
        print("-" * 40)
        print(f"[*] Đang thực thi phương pháp: {method.upper()}")
        
        # Chuyển đổi phương pháp
        detector.set_method(method)
        
        # Cho dataset accept service (chạy qua pipeline)
        dataset.accept(detector)
        
        # Trích xuất danh sách các index khả nghi
        anomalies_indices = detector.get_anomaly_candidates()
        print(f"   -> Đã tìm thấy {len(anomalies_indices)} điểm dị thường.")
        
        # Lấy boolean mask
        mask = detector.get_anomaly_mask()
        all_masks_dict[method] = mask
        
        # Vẽ biểu đồ riêng cho phương pháp này
        try:
            print(f"   -> Đang vẽ biểu đồ cho {method.upper()}...")
            plot_anomalies_single_method(dataset=dataset, anomaly_mask=mask, method_name=method.upper())
        except Exception as e:
            print(f"[LỖI] Không thể vẽ biểu đồ cho {method}: {e}")

    # 4. Vẽ biểu đồ so sánh tổng hợp
        print("-" * 40)
    print("[*] Đang vẽ biểu đồ so sánh tổng hợp tất cả các phương pháp...")
    try:
        plot_anomalies_all_methods(dataset=dataset, anomalies_dict=all_masks_dict)
        print("[THÀNH CÔNG] Pipeline hoàn tất!")
    except Exception as e:
        print(f"[LỖI] Không thể vẽ biểu đồ so sánh: {e}")

if __name__ == "__main__":
    main()