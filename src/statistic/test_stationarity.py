import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
from core.test_base import StationarityTesting

class ADFTesting(StationarityTesting):
    def __init__(self, column_name: str, alpha: float = 0.05):
        """Khởi tạo kiểm định tính dừng Augmented Dickey-Fuller (ADF)."""
        super().__init__(column_name, alpha)
        self.step_name = "Augmented Dickey-Fuller (ADF) Test"

    def visitTableDataset(self, obj):
        """Thực hiện kiểm định ADF trên cột dữ liệu chỉ định."""
        self.dataset_name = getattr(obj, '_folder_path', "TimeSeriesDataset")
        try:
            series = obj.data[self.column_name].dropna()
            result = adfuller(series, autolag='AIC')
            self.p_value = result[1]
            self.is_stationary = self.p_value < self.alpha
            self.status = "Success"
        except Exception as e:
            self.status = f"Failed ({str(e)})"
        finally:
            self.log()

class KPSSTesting(StationarityTesting):
    def __init__(self, column_name: str, alpha: float = 0.05):
        """Khởi tạo kiểm định tính dừng KPSS."""
        super().__init__(column_name, alpha)
        self.step_name = "KPSS Test"

    def visitTableDataset(self, obj):
        """Thực hiện kiểm định KPSS trên cột dữ liệu chỉ định."""
        self.dataset_name = getattr(obj, '_folder_path', "TimeSeriesDataset")
        try:
            series = obj.data[self.column_name].dropna()
            result = kpss(series, regression='c', nlags="auto")
            self.p_value = result[1]
            self.is_stationary = self.p_value >= self.alpha
            self.status = "Success"
        except Exception as e:
            self.status = f"Failed ({str(e)})"
        finally:
            self.log()

class PPTesting(StationarityTesting):
    def __init__(self, column_name: str, alpha: float = 0.05):
        """Khởi tạo kiểm định tính dừng Phillips-Perron (PP)."""
        super().__init__(column_name, alpha)
        self.step_name = "Phillips-Perron (PP) Test"

    def visitTableDataset(self, obj):
        """Thực hiện kiểm định PP trên cột dữ liệu chỉ định."""
        self.dataset_name = getattr(obj, '_folder_path', "TimeSeriesDataset")
        try:
            series = obj.data[self.column_name].dropna()
            pp = PhillipsPerron(series)
            self.p_value = pp.pvalue
            self.is_stationary = self.p_value < self.alpha
            self.status = "Success"
        except Exception as e:
            self.status = f"Failed ({str(e)})"
        finally:
            self.log()