import numpy as np
from scipy import stats
from core.service_base import Visualization
from core.data_base import Object
from visualization.distribution import plot_distribution
from table.dataset import TableDataset

class NormalityTest(Visualization):
    def __init__(self, column_name: str, alpha: float = 0.05):
        self.step_name = "Base Normality Test"
        self.dataset_name = "Unknown"
        self.status = "Pending"
        self.column_name = column_name
        self.alpha = alpha

    def log(self):
        """Hàm log mặc định quản lý trạng thái của Pipeline"""
        print(f"Bước xử lý  : {self.step_name}")
        print(f"Tập dữ liệu : {self.dataset_name}")
        print(f"Thuộc tính  : {self.column_name}")
        print(f"Trạng thái  : {self.status}\n")

    def log_explain(self, stat: float, p_value: float):
        """Hàm báo cáo kết quả kiểm định và giải thích thực nghiệm"""
        print(f"\n--- BÁO CÁO KIỂM ĐỊNH {self.step_name.upper()} ---")
        print(f"[*] Giả thuyết H0: Dữ liệu '{self.column_name}' tuân theo phân phối chuẩn.")
        print(f"[*] Thống kê (Stat): {stat:.4f} | p-Value: {p_value:.4e} | Alpha: {self.alpha}")
        
        if p_value > self.alpha:
            print("[*] Kết luận: CHẤP NHẬN H0.")
            print("[*] Thực nghiệm: p-value > alpha. Trực quan cho thấy đồ thị KDE có hình quả chuông đối xứng. "
                  "Không có bằng chứng vi phạm phân phối chuẩn.")
        else:
            print("[*] Kết luận: BÁC BỎ H0.")
            print("[*] Thực nghiệm: p-value cực nhỏ. Trực quan đồ thị KDE sẽ cho thấy dữ liệu bị lệch (skewness), "
                  "có đuôi dài, hoặc đỉnh quá nhọn/tù (kurtosis) khác xa phân phối chuẩn.")
        print("-" * 50)
class ShapiroTest(NormalityTest):
    def __init__(self, column_name: str, alpha: float = 0.05):
        super().__init__(column_name, alpha)
        self.step_name = "Shapiro-Wilk Test"

    def run(self, obj: Object):
        self.dataset_name = getattr(obj, '_file_path', "TableDataset")
        try:
            data = obj.data[self.column_name].dropna().values
            
            # Shapiro nhạy cảm với tập dữ liệu lớn
            if len(data) > 5000:
                data = np.random.choice(data, 5000, replace=False)

            plot_distribution(data, self.column_name, self.step_name)
            
            stat, p_value = stats.shapiro(data)
            
            # Chỉ việc gọi hàm từ class cha
            self.log_explain(stat, p_value)
            self.status = "Success"
            
        except Exception as e:
            self.status = f"Failed ({str(e)})"
        finally:
            self.log()


class DPearsonTest(NormalityTest):
    def __init__(self, column_name: str, alpha: float = 0.05):
        super().__init__(column_name, alpha)
        self.step_name = "D'Agostino-Pearson Test"

    def run(self, obj: Object):
        self.dataset_name = getattr(obj, '_file_path', "TableDataset")
        try:
            data = obj.data[self.column_name].dropna().values

            if len(data) < 8:
                raise ValueError("Cần tối thiểu 8 giá trị.")

            plot_distribution(data, self.column_name, self.step_name)
            
            stat, p_value = stats.normaltest(data)
            
            # Chỉ việc gọi hàm từ class cha
            self.log_explain(stat, p_value)
            self.status = "Success"
            
        except Exception as e:
            self.status = f"Failed ({str(e)})"
        finally:
            self.log()