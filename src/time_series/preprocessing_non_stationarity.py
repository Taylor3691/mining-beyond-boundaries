import numpy as np
import pandas as pd
from scipy.stats import boxcox
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
from core.test_base import StationarityTesting
from core.service_base import Preprocessing

class ADFTesting(StationarityTesting):
    def __init__(self, column_name: str, alpha: float = 0.05):
        super().__init__(column_name, alpha)
        self.step_name = "Augmented Dickey-Fuller (ADF) Test"

    def visitTableDataset(self, obj):
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
        super().__init__(column_name, alpha)
        self.step_name = "KPSS Test"

    def visitTableDataset(self, obj):
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
        super().__init__(column_name, alpha)
        self.step_name = "Phillips-Perron (PP) Test"

    def visitTableDataset(self, obj):
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

class StationarityTransformer(Preprocessing):
    def __init__(self, column_name: str, method: str = 'diff_1'):
        self.column_name = column_name
        self.method = method
        self.step_name = f"Stationarity Transform ({method})"
        self.status = "Pending"
        self._lambda = None

    def fit(self, df: pd.DataFrame):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        series = df_out[self.column_name]

        if self.method == 'diff_1':
            df_out[f"{self.column_name}_diff_1"] = series.diff()
        elif self.method == 'diff_2':
            df_out[f"{self.column_name}_diff_2"] = series.diff().diff()
        elif self.method == 'log':
            offset = abs(series.min()) + 1 if series.min() <= 0 else 0
            df_out[f"{self.column_name}_log"] = np.log(series + offset)
        elif self.method == 'boxcox':
            offset = abs(series.min()) + 1 if series.min() <= 0 else 0
            transformed, self._lambda = boxcox(series + offset)
            df_out[f"{self.column_name}_boxcox"] = transformed
        else:
            raise ValueError("Method không hợp lệ.")
            
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def visitImageDataset(self, obj):
        pass

    def run(self, obj):
        try:
            if hasattr(obj, 'data'):
                obj.data = self.fit_transform(obj.data)
                self.status = "Success"
            else:
                self.status = "Failed"
        except Exception as e:
            self.status = f"Failed ({str(e)})"
        finally:
            self.log()

    def log(self):
        print(f"Bước xử lý : {self.step_name} | Trạng thái: {self.status}")