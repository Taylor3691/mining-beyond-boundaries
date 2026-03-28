from core import DistributionTesting
from image import ImageDataset
import numpy as np
from scipy import stats

class KolmogorovSmirnovTesting(DistributionTesting):
    def __init__(self, alpha=0.05):
        self.step_name = "Distribution Consistency Check"
        self.test_name = "Kolmogorov-Smirnov Test"
        self.alpha = alpha
        self.h0_hypothesis = "Hai tập dữ liệu (Gốc và Đã xử lý) có cùng phân phối xác suất."
        
        self.statistic = None
        self.p_value = None
        self.conclusion = ""
        self.is_rejected = False

    def visitImageDataset(self, obj: ImageDataset):
        # check attribute, nếu không có thì raise lỗi
        if not hasattr(obj, '_origin_images') or not hasattr(obj, '_processed_images'):
            raise AttributeError("ImageDataset phải có thuộc tính '_origin_images' và '_processed_images'")

        # flatten ảnh
        try:
            data_orig_flat = np.concatenate(obj._origin_images).flatten()
            data_proc_flat = np.concatenate(obj._processed_images).flatten()
        except Exception as e:
            raise ValueError(f"Lỗi khi xử lý mảng ảnh: {e}")

        # test
        self.test(data_orig_flat, data_proc_flat)
        return
    
    def run(self, obj: ImageDataset):
        if isinstance(obj, ImageDataset):
            self.visitImageDataset(obj)
        return 
    
    def log(self):
        print(f"Step: {self.step_name}")
        print(f"Kiểm định: {self.test_name}")
        print(f"Giả thuyết H0: {self.h0_hypothesis}")
        
        print(f"P-value: {self.p_value:.10f}")
        print(f"Statistic: {self.statistic:.6f}")
            
        if self.is_rejected:
            print(f"Kết luận: {self.conclusion}.")
            print(f"Thống kê cho thấy việc xử lý ảnh đã làm thay đổi bản chất phân phối dữ liệu (p <= {self.alpha}).")
        else:
            print(f"Kết luận: {self.conclusion}.")
            print(f"{self.h0_hypothesis}")
        return
            
    def test(self, data_orig: np.ndarray, data_proc: np.ndarray):
        # stats.ks_2samp trả về (ks_statistic, p_value)
        self.statistic, self.p_value = stats.ks_2samp(data_orig, data_proc)
        
        # Đánh giá dựa trên alpha
        self.is_rejected = self.p_value <= self.alpha
        if self.is_rejected:
            self.conclusion = "Bác bỏ giả thuyết H0"
        else:
            self.conclusion = "Chấp nhận giả thuyết H0"
        return self.statistic, self.p_value
