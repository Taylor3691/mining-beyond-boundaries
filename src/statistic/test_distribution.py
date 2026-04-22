from core import DistributionTesting
from image import ImageDataset
import numpy as np
from scipy import stats
from IPython.display import display, Markdown
from table.dataset import TableDataset  

class KolmogorovSmirnovTesting(DistributionTesting):
    def __init__(self, alpha=0.05):
        """Khởi tạo kiểm định Kolmogorov-Smirnov cho hai mẫu."""
        self.step_name = "Distribution Consistency Check"
        self.test_name = "Kolmogorov-Smirnov Test"
        self.alpha = alpha
        self.h0_hypothesis = "Hai tập dữ liệu (Gốc và Đã xử lý) có cùng phân phối xác suất."
        
        self.statistic = None
        self.p_value = None
        self.conclusion = ""
        self.is_rejected = False

    def visitImageDataset(self, obj: ImageDataset):
        """Kiểm định phân phối cho dữ liệu hình ảnh."""
        if not hasattr(obj, '_origin_images') or not hasattr(obj, '_processed_images'):
            raise AttributeError("ImageDataset phải có thuộc tính '_origin_images' và '_processed_images'")

        try:
            data_orig_flat = np.concatenate(obj._origin_images).flatten()
            data_proc_flat = np.concatenate(obj._processed_images).flatten()
        except Exception as e:
            raise ValueError(f"Lỗi khi xử lý mảng ảnh: {e}")

        self.test(data_orig_flat, data_proc_flat)
        return
    
    def visitTableDataset(self, obj):
        """Kiểm định phân phối cho dữ liệu bảng."""
        if not hasattr(obj, '_origin_data') or getattr(obj, '_origin_data') is None:
             raise AttributeError("TableDataset phải có thuộc tính '_origin_data' để thực hiện KS Test. Hãy cập nhật TableDataset.")
        
        try:
            orig_numeric = obj._origin_data.select_dtypes(include=[np.number]).values.flatten()
            proc_numeric = obj.data.select_dtypes(include=[np.number]).values.flatten()
            
            data_orig_flat = orig_numeric[~np.isnan(orig_numeric)]
            data_proc_flat = proc_numeric[~np.isnan(proc_numeric)]
        except Exception as e:
            raise ValueError(f"Lỗi khi xử lý dữ liệu bảng: {e}")

        self.test(data_orig_flat, data_proc_flat)
        return
    
    def run(self, obj):
        """Thực thi kiểm định dựa trên kiểu dữ liệu của đối tượng."""
        obj_type = type(obj).__name__
        if obj_type == "ImageDataset":
            self.visitImageDataset(obj)
        elif obj_type == "TableDataset":
            self.visitTableDataset(obj)
        else:
            raise ValueError(f"Lớp {self.__class__.__name__} chưa hỗ trợ type: {obj_type}")
        return
    
    def log(self):
        """In báo cáo kết quả kiểm định ra màn hình."""
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
        """Thực hiện tính toán chỉ số thống kê KS."""
        self.statistic, self.p_value = stats.ks_2samp(data_orig, data_proc)
        
        self.is_rejected = self.p_value <= self.alpha
        if self.is_rejected:
            self.conclusion = "Bác bỏ giả thuyết H0"
        else:
            self.conclusion = "Chấp nhận giả thuyết H0"
        return self.statistic, self.p_value

class ShapiroWilkTesting(DistributionTesting):
    def __init__(self, column_name: str, alpha: float = 0.05):
        """Khởi tạo kiểm định Shapiro-Wilk cho phân phối chuẩn."""
        super().__init__()
        self._step_name = "Shapiro-Wilk Normality Test"
        self._column_name = column_name
        self._alpha = alpha
        self._stat = None
        self._p_value = None
        self._is_normal = False

    def run(self):
        """Thực thi toàn bộ quy trình kiểm định."""
        if self._dataset is None:
            self._status = "Failed — Dataset is None"
            return
            
        self.visitTableDataset(self._dataset)
        
        if self._status == "Success":
            self.test()

    def log(self):
        """In trạng thái thực thi kiểm định."""
        print("=" * 55)
        print(f"Step    : {self._step_name}")
        print(f"Column  : {self._column_name}")
        print(f"Dataset : {self._dataset._folder_path if self._dataset and self._dataset._folder_path else 'None'}")
        print(f"Status  : {self._status}")
        print("=" * 55)

    def visitTableDataset(self, obj: TableDataset):
        """Kiểm tra sự tồn tại của dữ liệu và cột cần phân tích."""
        try:
            self._dataset = obj
            if obj.data is None or obj.data.empty:
                self._status = "Failed — Dữ liệu trống."
                return
                
            if self._column_name not in obj.columns:
                self._status = f"Failed — Không tìm thấy cột '{self._column_name}'."
                return
                
            self._status = "Success"
        except Exception as e:
            self._status = f"Failed in visitTableDataset() — {str(e)}"

    def test(self):
        """Thực hiện kiểm định Shapiro-Wilk trên cột dữ liệu."""
        try:
            data = self._dataset.data[self._column_name].dropna().values
            
            if len(data) < 3:
                self._status = "Failed — Cần tối thiểu 3 mẫu để chạy Shapiro-Wilk."
                return

            if len(data) > 5000:
                np.random.seed(42)
                data = np.random.choice(data, 5000, replace=False)

            self._stat, self._p_value = stats.shapiro(data)
            self._is_normal = self._p_value > self._alpha
            self._analyze()
            self._status = "Success"
            
        except Exception as e:
            self._status = f"Failed in test() — {str(e)}"

    def _analyze(self):
        """Phân tích kết quả và hiển thị báo cáo chi tiết."""
        conclusion = "**CHẤP NHẬN** giả thuyết H0 (Dữ liệu tuân theo phân phối chuẩn)." if self._is_normal else "**BÁC BỎ** giả thuyết H0 (Dữ liệu KHÔNG tuân theo phân phối chuẩn)."
        
        practical_note = ""
        if self._is_normal:
             practical_note = "Dữ liệu biến động tự nhiên, tính ngẫu nhiên cao và không bị thao túng bởi các yếu tố bên ngoài hay các ngưỡng quy định cứng nhắc. Các mẫu lấy được đại diện tốt cho quần thể."
        else:
             practical_note = "Dữ liệu **không biến động tự nhiên**. Đối chiếu với bối cảnh thực tế của hồ sơ cấp phép xây dựng (Building Permits), sự méo lệch này chứng tỏ sự can thiệp của con người và quy định. *Ví dụ:* Các chỉ số như diện tích hoặc chi phí thường bị gom cụm tại các mức quy định (để lách luật hoặc vừa đủ chuẩn), hoặc bị giới hạn bởi quy hoạch đô thị trần/sàn, tạo ra nhiều điểm dị biệt (outliers) bẻ gãy quy luật hình chuông tự nhiên."

        md = f"""
## Nhận xét: Kiểm định Phân phối chuẩn Shapiro-Wilk
**Thuộc tính phân tích:** `{self._column_name}`

### 1. Kết quả Thống kê
- **Giả thuyết H0:** Dữ liệu của biến `{self._column_name}` có phân phối chuẩn.
- **Giá trị Thống kê (W):** `{self._stat:.4f}`
- **p-Value:** `{self._p_value:.4e}`
- **Mức ý nghĩa (Alpha):** `{self._alpha}`

### 2. Kết luận
- Mức p-Value là `{self._p_value:.4e}` {'**lớn**' if self._is_normal else '**nhỏ**'} hơn Alpha.
-  {conclusion}

### 3. Phân tích Thực tiễn
- {practical_note}
"""
        display(Markdown(md))


class AgostinoPearsonTesting(DistributionTesting):
    def __init__(self, column_name: str, alpha: float = 0.05):
        """Khởi tạo kiểm định D'Agostino-Pearson cho phân phối chuẩn."""
        super().__init__()
        self._step_name = "D'Agostino-Pearson Normality Test"
        self._column_name = column_name
        self._alpha = alpha
        self._stat = None
        self._p_value = None
        self._is_normal = False

    def run(self):
        """Thực thi toàn bộ quy trình kiểm định."""
        if self._dataset is None:
            self._status = "Failed — Dataset is None"
            return
        self.visitTableDataset(self._dataset)
        if self._status == "Success":
            self.test()

    def log(self):
        """In trạng thái thực thi kiểm định."""
        print("=" * 55)
        print(f"Step    : {self._step_name}")
        print(f"Column  : {self._column_name}")
        print(f"Dataset : {self._dataset._folder_path if self._dataset and self._dataset._folder_path else 'None'}")
        print(f"Status  : {self._status}")
        print("=" * 55)

    def visitTableDataset(self, obj: TableDataset):
        """Kiểm tra sự tồn tại của dữ liệu và cột cần phân tích."""
        try:
            self._dataset = obj
            if obj.data is None or obj.data.empty:
                self._status = "Failed — Dữ liệu trống."
                return
                
            if self._column_name not in obj.columns:
                self._status = f"Failed — Không tìm thấy cột '{self._column_name}'."
                return
                
            self._status = "Success"
        except Exception as e:
            self._status = f"Failed in visitTableDataset() — {str(e)}"

    def test(self):
        """Thực hiện kiểm định D'Agostino-Pearson trên cột dữ liệu."""
        try:
            data = self._dataset.data[self._column_name].dropna().values
            
            if len(data) < 8:
                self._status = "Failed — D'Agostino-Pearson yêu cầu tối thiểu 8 giá trị hợp lệ."
                return

            self._stat, self._p_value = stats.normaltest(data)
            self._is_normal = self._p_value > self._alpha
            self._analyze()
            self._status = "Success"
            
        except Exception as e:
            self._status = f"Failed in test() — {str(e)}"

    def _analyze(self):
        """Phân tích kết quả và hiển thị báo cáo chi tiết."""
        conclusion = "**CHẤP NHẬN** giả thuyết H0 (Dữ liệu tuân theo phân phối chuẩn)." if self._is_normal else "**BÁC BỎ** giả thuyết H0 (Dữ liệu KHÔNG tuân theo phân phối chuẩn)."
        
        practical_note = ""
        if self._is_normal:
             practical_note = "Phân phối của dữ liệu có độ lệch (Skewness) và độ nhọn (Kurtosis) cân bằng, nằm trong ngưỡng an toàn của quy luật xác suất chuẩn."
        else:
             practical_note = "Kiểm định này cực kỳ nhạy với mức độ 'lệch' và 'nhọn' của biểu đồ. Việc thất bại kiểm định này chỉ ra rằng dữ liệu có hiện tượng **đuôi quá dày** (chứa quá nhiều hồ sơ siêu lớn/siêu nhỏ) hoặc **tập trung quá mức** vào một đỉnh duy nhất (ví dụ: một mức phí chuẩn mà 80% nhà thầu đều khai báo giống hệt nhau). Điều này phản ánh rõ nét yếu tố hành vi thị trường thay vì các số đo biến thiên tự nhiên."

        md = f"""
## Nhận xét: Kiểm định D'Agostino-Pearson K-squared
**Thuộc tính phân tích:** `{self._column_name}`

### 1. Kết quả Thống kê
- **Giả thuyết H0:** Dữ liệu của biến `{self._column_name}` có phân phối chuẩn.
- **Giá trị Thống kê (K²):** `{self._stat:.4f}`
- **p-Value:** `{self._p_value:.4e}`
- **Mức ý nghĩa (Alpha):** `{self._alpha}`

### 2. Kết luận
- Mức p-Value là `{self._p_value:.4e}` {'**lớn**' if self._is_normal else '**nhỏ**'} hơn Alpha.
- {conclusion}

### 3. Phân tích Thực tiễn 
- {practical_note}
"""
        display(Markdown(md))
