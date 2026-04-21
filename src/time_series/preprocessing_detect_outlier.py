import numpy as np
import pandas as pd
from typing import Any, Dict, List
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL
from core.service_base import Preprocessing
import os

class DetectOutlierTimeSeries(Preprocessing):
    def __init__(self,
                 method: str = 'z-score',
                 window_size: int = 7,  # Chu kỳ 7 ngày cho Covid-19
                 threshold: float = 3.0, # Ngưỡng Z-score
                 contamination: Any = 0.05 # Cho phép float hoặc 'auto'
                 ):
        
        self._supported_methods = ['z-score', 'iforest', 'stl']
        if method not in self._supported_methods:
            raise ValueError(f"Method '{method}' not supported. Choose from {self._supported_methods}")
        
        self._method = method
        self._window_size = window_size
        self._threshold = threshold
        self._contamination = contamination
        
        self._stats: Dict[str, Any] = {}
        self._anomalies_mask = None 
        self._anomaly_indices: List[int] = [] 
        return
    
    # Getters & Setters
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
        print(f"[*] Đã chuyển phương pháp phát hiện dị thường sang: {self._method.upper()}")

    def get_anomaly_candidates(self) -> List[int]:
        if not self._anomaly_indices:
            print("[WARNING] Chưa có điểm dị thường nào. Hãy đảm bảo đã chạy fit() hoặc run().")
        return self._anomaly_indices

    def get_anomaly_mask(self) -> np.ndarray:
        return self._anomalies_mask

    # Core Logic
    def fit(self, series: Any):
        """Tính toán và tìm các điểm khả nghi dựa trên phương pháp đã chọn"""
        
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        self._anomalies_mask = np.zeros(len(series), dtype=bool)

        # Xử lý missing values
        series_clean = series.bfill().ffill()

        # Route theo phương pháp
        if self._method == 'z-score':
            self._fit_zscore_deseasonalized(series_clean)
        elif self._method == 'iforest':
            self._fit_isolation_forest(series_clean)
        elif self._method == 'stl':
            self._fit_stl_thresholding(series_clean)
            
        if isinstance(self._anomalies_mask, pd.Series):
            self._anomalies_mask = self._anomalies_mask.to_numpy()

        # Cập nhật danh sách ứng viên
        self._anomaly_indices = np.where(self._anomalies_mask)[0].tolist()

    def transform(self, arr: np.ndarray):
        pass

    def fit_transform(self, arr: np.ndarray):
        pass

    def run(self, obj):
        if obj.__class__.__name__ == "TimeSeriesDataset":
            self.visitTimeSeriesDataset(obj)
        else:
            print(f"[CẢNH BÁO] DetectOutlierTimeSeries không hỗ trợ loại dữ liệu: {type(obj)}")
        return

    def visitTimeSeriesDataset(self, obj):
        if getattr(obj, 'target', None) is None:
            raise ValueError("Dataset chưa được gọi hàm set_target().")
        
        try:
            target_series = obj.target
            self.fit(target_series)
            
            obj._anomaly_mask = self._anomalies_mask 
            
        except Exception as e:
            import traceback
            print(f"\n[ERROR] Thuật toán {self._method.upper()} gặp sự cố:")
            traceback.print_exc()
        finally:
            self.log()
        return

    def visitImageDataset(self, obj):
        pass

    def log(self):
        pass

    # Implementations cho từng thuật toán
    def _fit_zscore_deseasonalized(self, series: pd.Series):
        stl = STL(series, period=self._window_size, robust=True)
        res = stl.fit()
        
        deseasonalized = series - res.seasonal
        
        mean_val = deseasonalized.mean()
        std_val = deseasonalized.std()
        
        z_scores = (deseasonalized - mean_val) / (std_val + 1e-8)
        self._anomalies_mask = np.abs(z_scores) > self._threshold
        
        self._stats['mean'] = mean_val
        self._stats['std'] = std_val

    def _fit_isolation_forest(self, series: pd.Series):
        df_roll = pd.DataFrame({'y': series})
        for i in range(1, self._window_size):
            df_roll[f'y_lag_{i}'] = df_roll['y'].shift(i)
        
        df_roll = df_roll.bfill().ffill()
        X = df_roll.values

        # Khởi tạo model
        model = IsolationForest(contamination=self._contamination, random_state=42)
        preds = model.fit_predict(X)
        
        self._anomalies_mask = (preds == -1)

    def _fit_stl_thresholding(self, series: pd.Series):
        stl = STL(series, period=self._window_size, robust=True)
        res = stl.fit()
        resid = res.resid
        
        q1 = np.nanpercentile(resid, 25)
        q3 = np.nanpercentile(resid, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        self._anomalies_mask = (resid < lower_bound) | (resid > upper_bound)
        self._stats['lower_bound'] = lower_bound
        self._stats['upper_bound'] = upper_bound