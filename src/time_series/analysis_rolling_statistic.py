import pandas as pd
from core.service_base import Visualization
from visualization.comparison import plot_rolling_statistics

class RollingStatisticAnalysis(Visualization):
    """Phân tích các chỉ số thống kê trượt (Rolling Statistics) của chuỗi thời gian."""

    def __init__(self, region: str = "World", feature_name: str = "Confirmed", stat_type: str = "Mean"):
        """
        Khởi tạo bộ phân tích thống kê trượt.

        Input:
            region: Tên khu vực khảo sát.
            feature_name: Tên biến khảo sát.
            stat_type: Loại thống kê trượt ('Mean' hoặc 'STD').
        """
        self.region = region
        self.feature_name = feature_name
        
        if stat_type not in ["Mean", "STD", "mean", "std"]:
            raise ValueError("[ERROR] Tham số stat_type phải là 'Mean' hoặc 'STD'.")
        self.stat_type = stat_type

    def run(self, dataset):
        """Thực thi tính toán và vẽ biểu đồ thống kê trượt trên dataset."""
        df = dataset.data
        time_col = dataset._time_column
        
        if self.region != "World":
            df_filtered = df[df['Country/Region'] == self.region]
        else:
            df_filtered = df

        df_grouped = df_filtered.groupby(time_col)[self.feature_name].sum().reset_index()

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