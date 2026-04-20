import pandas as pd
from core.service_base import Visualization
from visualization.comparison import plot_rolling_statistics

class RollingStatisticAnalysis(Visualization):
    # Nhận tham số cấu hình: biến cần tính và giá trị thống kê ("Mean" hoặc "STD")
    def __init__(self, region: str = "World", feature_name: str = "Confirmed", stat_type: str = "Mean"):
        self.region = region
        self.feature_name = feature_name
        
        if stat_type not in ["Mean", "STD", "mean", "std"]:
            raise ValueError("[ERROR] Tham số stat_type phải là 'Mean' hoặc 'STD'.")
        self.stat_type = stat_type

    def run(self, dataset):
        df = dataset.data
        time_col = dataset._time_column
        
        # 1. Lọc khu vực
        if self.region != "World":
            df_filtered = df[df['Country/Region'] == self.region]
        else:
            df_filtered = df

        # 2. Gom nhóm theo ngày và tính tổng
        df_grouped = df_filtered.groupby(time_col)[self.feature_name].sum().reset_index()

        # 3. Trực quan hóa 4 đường bằng hàm bên comparison.py
        print(f"[EXECUTE] Đang vẽ Rolling {self.stat_type.upper()} cho biến '{self.feature_name}'...")
        plot_rolling_statistics(
            dates=df_grouped[time_col], 
            values=df_grouped[self.feature_name], 
            feature_name=self.feature_name,
            stat_type=self.stat_type,
            region=self.region
        )

    # Implement các hàm abstract
    def log(self): pass
    def visitImageDataset(self, dataset=None): pass