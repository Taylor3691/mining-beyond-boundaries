import pandas as pd
from core.service_base import Visualization
from visualization.comparison import plot_time_series

class AnalysisDataTimeComponents(Visualization):
    """Phân tích các thành phần dữ liệu theo thời gian (Time Plot)."""

    def __init__(self, region: str = "World", feature_name: str = "Confirmed"):
        """
        Khởi tạo bộ phân tích thành phần thời gian.

        Input:
            region: Tên khu vực khảo sát.
            feature_name: Tên biến khảo sát.
        """
        self.region = region
        self.feature_name = feature_name

    def change_region(self, new_region: str):
        """
        Thay đổi khu vực khảo sát.

        Input:
            new_region: Tên khu vực mới.

        Output:
            None.
        """
        self.region = new_region
        print(f"[INFO] Đã chuyển khu vực khảo sát sang: {self.region}")

    def change_feature(self, new_feature: str, dataset_columns: list):
        """
        Thay đổi biến khảo sát.

        Input:
            new_feature: Tên biến mới.
            dataset_columns: Danh sách các cột hợp lệ trong dataset.

        Output:
            None.
        """
        if new_feature not in dataset_columns:
            raise ValueError(f"[ERROR] Biến '{new_feature}' không tồn tại trong Dataset!")
        self.feature_name = new_feature
        print(f"[INFO] Đã chuyển biến khảo sát sang: {self.feature_name}")

    def run(self, dataset):
        """
        Trích xuất dữ liệu và vẽ biểu đồ chuỗi thời gian (Time Plot).

        Input:
            dataset: Đối tượng TimeSeriesDataset chứa dữ liệu.

        Output:
            None (hiển thị biểu đồ).
        """
        df = dataset.data
        time_col = dataset._time_column
        
        if self.feature_name not in df.columns:
            raise ValueError(f"[ERROR] Không tìm thấy cột '{self.feature_name}'")

        if self.region != "World":
            if 'Country/Region' not in df.columns:
                raise ValueError("[ERROR] Dữ liệu không có cột 'Country/Region' để lọc!")
            df_filtered = df[df['Country/Region'] == self.region]
            if df_filtered.empty:
                raise ValueError(f"[ERROR] Không có dữ liệu cho khu vực '{self.region}'")
        else:
            df_filtered = df # Lấy toàn thế giới

        df_grouped = df_filtered.groupby(time_col)[self.feature_name].sum().reset_index()

        print(f"[EXECUTE] Đang vẽ Time Plot cho biến '{self.feature_name}' tại '{self.region}'...")
        plot_time_series(
            dates=df_grouped[time_col], 
            values=df_grouped[self.feature_name], 
            feature_name=self.feature_name, 
            region=self.region
        )

    def log(self):
        pass

    def visitImageDataset(self, dataset=None):
        pass