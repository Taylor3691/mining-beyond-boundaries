import numpy as np
import pandas as pd
from scipy.stats import boxcox
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
from core.test_base import StationarityTesting
from core.service_base import Preprocessing

class ADFTesting(StationarityTesting):
    """Kiểm định tính dừng ADF cho chuỗi thời gian."""

    def __init__(self, column_name: str, alpha: float = 0.05):
        """
        Khởi tạo kiểm định ADF.

        Input:
            column_name: Tên cột cần kiểm định.
            alpha: Mức ý nghĩa thống kê (mặc định 0.05).

        Output:
            None.
        """
        super().__init__(column_name, alpha)
        self.step_name = "Augmented Dickey-Fuller (ADF) Test"

    def visitTableDataset(self, obj):
        """
        Thực hiện kiểm định ADF trên cột dữ liệu chỉ định.

        Input:
            obj: Đối tượng dataset có thuộc tính data (DataFrame).

        Output:
            None (cập nhật p_value và is_stationary).
        """
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
    """Kiểm định tính dừng KPSS cho chuỗi thời gian."""

    def __init__(self, column_name: str, alpha: float = 0.05):
        """
        Khởi tạo kiểm định KPSS.

        Input:
            column_name: Tên cột cần kiểm định.
            alpha: Mức ý nghĩa thống kê (mặc định 0.05).

        Output:
            None.
        """
        super().__init__(column_name, alpha)
        self.step_name = "KPSS Test"

    def visitTableDataset(self, obj):
        """
        Thực hiện kiểm định KPSS trên cột dữ liệu chỉ định.

        Input:
            obj: Đối tượng dataset có thuộc tính data (DataFrame).

        Output:
            None (cập nhật p_value và is_stationary).
        """
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
    """Kiểm định tính dừng Phillips-Perron cho chuỗi thời gian."""

    def __init__(self, column_name: str, alpha: float = 0.05):
        """
        Khởi tạo kiểm định Phillips-Perron.

        Input:
            column_name: Tên cột cần kiểm định.
            alpha: Mức ý nghĩa thống kê (mặc định 0.05).

        Output:
            None.
        """
        super().__init__(column_name, alpha)
        self.step_name = "Phillips-Perron (PP) Test"

    def visitTableDataset(self, obj):
        """
        Thực hiện kiểm định PP trên cột dữ liệu chỉ định.

        Input:
            obj: Đối tượng dataset có thuộc tính data (DataFrame).

        Output:
            None (cập nhật p_value và is_stationary).
        """
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
    """Biến đổi chuỗi thời gian để đạt tính dừng (differencing, log, boxcox)."""

    def __init__(self, column_name: str, method: str = 'diff_1'):
        """
        Khởi tạo bộ biến đổi tính dừng.

        Input:
            column_name: Tên cột cần biến đổi.
            method: Phương pháp biến đổi ('diff_1', 'diff_2', 'log', 'boxcox').

        Output:
            None.
        """
        self.column_name = column_name
        self.method = method
        self.step_name = f"Stationarity Transform ({method})"
        self.status = "Pending"
        self._lambda = None

    def fit(self, df: pd.DataFrame):
        """
        Không cần tính toán tham số (stateless transform).

        Input:
            df: DataFrame (không sử dụng).

        Output:
            None.
        """
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Áp dụng biến đổi tính dừng lên cột dữ liệu.

        Input:
            df: DataFrame chứa cột cần biến đổi.

        Output:
            pd.DataFrame: DataFrame mới với cột biến đổi được thêm vào.
        """
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
        """
        Kết hợp fit và transform.

        Input:
            df: DataFrame chứa cột cần biến đổi.

        Output:
            pd.DataFrame: DataFrame sau biến đổi.
        """
        self.fit(df)
        return self.transform(df)

    def visitImageDataset(self, obj):
        """
        Không hỗ trợ dữ liệu hình ảnh.

        Input:
            obj: Đối tượng ImageDataset.

        Output:
            None.
        """
        pass

    def run(self, obj):
        """
        Thực thi biến đổi tính dừng trên đối tượng dataset.

        Input:
            obj: Đối tượng dataset có thuộc tính data.

        Output:
            None (cập nhật obj.data).
        """
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
        """
        In trạng thái thực thi của bước biến đổi.

        Input:
            Không có.

        Output:
            None (in ra màn hình).
        """
        print(f"Bước xử lý : {self.step_name} | Trạng thái: {self.status}")