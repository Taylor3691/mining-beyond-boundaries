import pandas as pd
from core.service_base import Visualization
from visualization.distribution import plot_acf_chart, plot_pacf_chart

class PACFAnalysisStructure(Visualization):
    # Thêm 2 tham số: use_pacf (để chọn vẽ ACF hay PACF) và n_lags (số quan sát quá khứ)
    def __init__(self, region: str = "World", feature_name: str = "Confirmed", use_pacf: bool = False, n_lags: int = 40):
        self.region = region
        self.feature_name = feature_name
        self.use_pacf = use_pacf
        self.n_lags = n_lags

    def run(self, dataset):
        df = dataset.data
        time_col = dataset._time_column
        
        # 1. Lọc dữ liệu theo khu vực
        if self.region != "World":
            df_filtered = df[df['Country/Region'] == self.region]
        else:
            df_filtered = df

        # 2. Gom nhóm theo ngày và tính tổng
        df_grouped = df_filtered.groupby(time_col)[self.feature_name].sum().reset_index()
        series_data = df_grouped[self.feature_name]

        # 3. Trực quan hóa dựa trên biến use_pacf
        if self.use_pacf:
            print(f"[EXECUTE] Đang vẽ biểu đồ PACF cho biến '{self.feature_name}' (Lag = {self.n_lags})...")
            plot_pacf_chart(series_data, self.n_lags, title=f"PACF of {self.feature_name} in {self.region}")
        else:
            print(f"[EXECUTE] Đang vẽ biểu đồ ACF cho biến '{self.feature_name}' (Lag = {self.n_lags})...")
            plot_acf_chart(series_data, self.n_lags, title=f"ACF of {self.feature_name} in {self.region}")

    # Các hàm abstract bắt buộc
    def log(self): pass
    def visitImageDataset(self, dataset=None): pass