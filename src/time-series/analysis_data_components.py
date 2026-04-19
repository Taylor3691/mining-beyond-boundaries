import pandas as pd
from core.service_base import Visualization
from visualization.comparison import plot_time_series

class AnalysisDataTimeComponents(Visualization):
    def __init__(self, region: str = "World", feature_name: str = "Confirmed"):
        self.region = region
        self.feature_name = feature_name

    def change_region(self, new_region: str):
        """Hàm đổi khu vực khảo sát"""
        self.region = new_region
        print(f"[INFO] Đã chuyển khu vực khảo sát sang: {self.region}")

    def change_feature(self, new_feature: str, dataset_columns: list):
        """Hàm đổi tên biến. Nếu không tồn tại thì raise Error"""
        if new_feature not in dataset_columns:
            raise ValueError(f"[ERROR] Biến '{new_feature}' không tồn tại trong Dataset!")
        self.feature_name = new_feature
        print(f"[INFO] Đã chuyển biến khảo sát sang: {self.feature_name}")

    def run(self, dataset):
        """
        Logic thực thi việc trích xuất dữ liệu và gọi hàm vẽ
        """
        df = dataset.data
        time_col = dataset._time_column
        
        # 1. Kiểm tra tính hợp lệ của biến
        if self.feature_name not in df.columns:
            raise ValueError(f"[ERROR] Không tìm thấy cột '{self.feature_name}'")

        # 2. Lọc dữ liệu theo khu vực (Region)
        if self.region != "World":
            if 'Country/Region' not in df.columns:
                raise ValueError("[ERROR] Dữ liệu không có cột 'Country/Region' để lọc!")
            df_filtered = df[df['Country/Region'] == self.region]
            if df_filtered.empty:
                raise ValueError(f"[ERROR] Không có dữ liệu cho khu vực '{self.region}'")
        else:
            df_filtered = df # Lấy toàn thế giới

        # 3. Gom nhóm theo ngày (Vì một nước có thể có nhiều tỉnh/bang được ghi nhận rời rạc)
        # Cộng tổng số ca của tất cả các tỉnh/bang trong cùng 1 ngày
        df_grouped = df_filtered.groupby(time_col)[self.feature_name].sum().reset_index()

        # 4. Trực quan hóa
        print(f"[EXECUTE] Đang vẽ Time Plot cho biến '{self.feature_name}' tại '{self.region}'...")
        plot_time_series(
            dates=df_grouped[time_col], 
            values=df_grouped[self.feature_name], 
            feature_name=self.feature_name, 
            region=self.region
        )

    # ==========================================
    # Cài đặt các hàm abstract bắt buộc để không bị lỗi
    # ==========================================
    def log(self):
        pass

    def visitImageDataset(self, dataset=None):
        pass