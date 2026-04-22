import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose, STL 
import matplotlib.pyplot as plt

class DecompositionAnalysis:
    """Lớp cơ sở thực hiện phân rã chuỗi thời gian (Time Series Decomposition)."""

    def __init__(self, data_series: pd.Series, period: int):
        """
        Khởi tạo bộ phân rã.

        Input:
            data_series: Chuỗi dữ liệu số cần phân rã.
            period: Chu kỳ mùa vụ (seasonal period).
        """
        self._series = data_series.dropna()
        self._period = period
        self.trend = None
        self.seasonal = None
        self.resid = None
        self.observed = self._series

    def run(self):
        pass

    def residual_variance_ratio(self) -> float:
        """Đánh giá chất lượng: Tỉ lệ phương sai của Residual / Observed. Càng nhỏ càng tốt."""
        if self.resid is None:
            raise ValueError("Chưa chạy phân rã (run)!")
        
        var_resid = np.var(self.resid.dropna())
        var_obs = np.var(self.observed.dropna())
        
        if var_obs == 0:
            return 0.0
        return float(var_resid / var_obs)

    def plot(self, title="Time Series Decomposition"):
        """Hàm trực quan hóa cơ bản"""
        if self.trend is None:
            print("Chưa có dữ liệu để vẽ.")
            return
            
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(self.observed, label='Observed', color='black')
        axes[0].legend(loc='upper left')
        axes[0].set_title(title)
        
        axes[1].plot(self.trend, label='Trend', color='blue')
        axes[1].legend(loc='upper left')
        
        axes[2].plot(self.seasonal, label='Seasonal', color='green')
        axes[2].legend(loc='upper left')
        
        axes[3].scatter(self.resid.index, self.resid, label='Residual', color='red', marker='.')
        axes[3].axhline(y=0 if getattr(self, '_model_type', 'additive') == 'additive' else 1, 
                        color='grey', linestyle='--')
        axes[3].legend(loc='upper left')
        
        plt.tight_layout()
        plt.show()

class AdditiveDecomposition(DecompositionAnalysis):
    """Phân rã chuỗi thời gian theo mô hình cộng (Additive Model)."""

    def __init__(self, data_series: pd.Series, period: int):
        super().__init__(data_series, period)
        self._model_type = 'additive'

    def run(self):
        result = seasonal_decompose(self.observed, model='additive', period=self._period)
        self.trend = result.trend
        self.seasonal = result.seasonal
        self.resid = result.resid

    def detrended(self):
        return self.observed - self.trend

    def deseasonalized(self):
        return self.observed - self.seasonal

    def residual(self):
        return self.observed - self.trend - self.seasonal


class MultiplicativeDecomposition(DecompositionAnalysis):
    def run(self):
        # Kỹ thuật Add-1 Smoothing để triệt tiêu số 0
        safe_series = self.observed + 1.0 
        result = seasonal_decompose(safe_series, model='multiplicative', period=self._period)
        self.trend = result.trend
        self.seasonal = result.seasonal
        self.resid = result.resid

    def detrended(self):
        return (self.observed + 1.0) / self.trend

    def deseasonalized(self):
        return (self.observed + 1.0) / self.seasonal

    def residual(self):
        return (self.observed + 1.0) / (self.trend * self.seasonal)

    def residual_variance_ratio(self) -> float:
        if self.trend is None or self.seasonal is None: 
            return 0.0
            
        # Tính số ca dự báo (Y_hat) của mô hình nhân
        # Vì trước đó đã cộng 1, ta trừ đi 1 để trả về số ca thực tế
        y_hat_thuc_te = (self.trend * self.seasonal) - 1.0
        
        # Phần dư tuyệt đối (Đo bằng số ca bệnh)
        absolute_resid = self.observed - y_hat_thuc_te
        
        var_resid = np.var(absolute_resid.dropna())
        var_obs = np.var(self.observed.dropna())
        
        return float(var_resid / var_obs) if var_obs != 0 else 0.0


class STLDecomposition(DecompositionAnalysis):
    """Phân rã chuỗi thời gian mạnh mẽ bằng phương pháp STL."""

    def __init__(self, data_series: pd.Series, period: int):
        super().__init__(data_series, period)
        self._model_type = 'additive' # STL bản chất là Additive

    def set_period(self, new_period: int):
        """Đổi tham số period để thử nghiệm nhanh"""
        self._period = new_period
        self.run() # Chạy lại phân rã ngay lập tức

    def run(self):
        stl = STL(self.observed, period=self._period, robust=True)
        result = stl.fit()
        self.trend = result.trend
        self.seasonal = result.seasonal
        self.resid = result.resid

    def detrended(self):
        return self.observed - self.trend

    def deseasonalized(self):
        return self.observed - self.seasonal

    def residual(self):
        return self.observed - self.trend - self.seasonal