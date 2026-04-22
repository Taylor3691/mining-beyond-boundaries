import pandas as pd
import numpy as np
import scipy.stats as stats
from IPython.display import display, Markdown
from core.testing import DistributionTesting
from table.dataset import TableDataset 

class MCARLittleTesting(DistributionTesting):
    def __init__(self, alpha: float = 0.05):
        """
        Khởi tạo kiểm định Little's MCAR cho dữ liệu thiếu.
        
        Input:
            alpha: Mức ý nghĩa thống kê dùng để đưa ra kết luận kiểm định.
        
        Output:
            None.
        """
        super().__init__()
        self._step_name = "Little's MCAR Test (Global Chi-Square)"
        self._alpha = alpha
        self._stat = None        # Giá trị Chi-square
        self._p_value = None
        self._df_chi2 = None     # Bậc tự do 
        self._is_mcar = False

    def run(self):
        """
        Thực thi toàn bộ quy trình kiểm định.
        
        Input:
            Không có.
        
        Output:
            None.
        """
        if self._dataset is None:
            self._status = "Failed — Dataset is None"
            return
            
        self.visitTableDataset(self._dataset)
        if self._status == "Success":
            self.test()

    def log(self):
        """
        In trạng thái thực thi kiểm định.
        
        Input:
            Không có.
        
        Output:
            None.
        """
        print("=" * 55)
        print(f"Step    : {self._step_name}")
        print(f"Dataset : {self._dataset._folder_path if self._dataset else 'None'}")
        print(f"Status  : {self._status}")
        print("=" * 55)

    def visitTableDataset(self, obj):
        """
        Gắn dữ liệu TableDataset vào bộ kiểm định.
        
        Input:
            obj: Đối tượng dữ liệu đầu vào cần thực thi kiểm định.
        
        Output:
            None.
        """
        try:
            self._dataset = obj
            self._status = "Success"
        except Exception as e:
            self._status = f"Failed — {str(e)}"

    def test(self):
        """
        Thực hiện Little's MCAR Test bằng phương pháp xấp xỉ Chi-Square.
        
        Input:
            Không có.
        
        Output:
            None.
        """
        try:
            # 1. Chỉ lấy dữ liệu số (Numeric) để tính toán ma trận
            df = self._dataset.data.select_dtypes(include=[np.number])
            if df.isnull().sum().sum() == 0:
                self._status = "Success — Dataset không có giá trị thiếu."
                return

            n_samples, n_vars = df.shape
            
            # 2. Ước lượng Grand Mean và Grand Covariance (Xấp xỉ thay vì dùng EM để tránh tràn bộ nhớ)
            grand_mean = df.mean()
            # Sinh ma trận hiệp phương sai, điền rỗng bằng 0 tạm thời để nghịch đảo không bị lỗi
            grand_cov = df.cov() 
            
            # Kiểm tra định thức để đảm bảo ma trận cov có thể nghịch đảo
            if np.linalg.det(grand_cov.fillna(0).values) == 0:
                self._status = "Failed — Ma trận Hiệp phương sai bị suy biến (Singular Matrix). Có đa cộng tuyến hoàn hảo."
                return
                
            inv_grand_cov = pd.DataFrame(
                np.linalg.pinv(grand_cov.values), 
                columns=grand_cov.columns, 
                index=grand_cov.index
            )

            # 3. Phân loại các Pattern (Mẫu thiếu)
            # Tạo chuỗi nhị phân: VD "010" nghĩa là cột 2 bị thiếu
            missing_patterns = df.isnull().astype(int).astype(str).apply(lambda x: ''.join(x), axis=1)
            df['Pattern'] = missing_patterns
            
            d2_sum = 0.0 # Tổng khoảng cách
            df_total = 0 # Tổng bậc tự do
            
            # 4. Tính toán cho từng Pattern
            for pattern, group in df.groupby('Pattern'):
                # Lấy các biến được quan sát (observed) trong pattern này
                obs_vars = [df.columns[i] for i, val in enumerate(pattern) if val == '0' and df.columns[i] != 'Pattern']
                n_obs = len(obs_vars)
                
                if n_obs == 0 or n_obs == n_vars:
                    continue
                    
                # Mean của nhóm pattern này
                group_mean = group[obs_vars].mean()
                
                # Khoảng cách giữa mean của nhóm và grand mean
                mean_diff = group_mean - grand_mean[obs_vars]
                
                # Ma trận hiệp phương sai con của các biến quan sát được
                sub_inv_cov = inv_grand_cov.loc[obs_vars, obs_vars]
                
                # Tính khoảng cách Mahalanobis có trọng số (Số lượng mẫu trong pattern * D2)
                d2 = len(group) * np.dot(np.dot(mean_diff.values, sub_inv_cov.values), mean_diff.values.T)
                
                d2_sum += d2
                df_total += n_obs
                
            # Trừ đi tổng số biến quan sát được ban đầu để ra bậc tự do chuẩn của Little
            df_total = df_total - n_vars

            # 5. Tính P-Value từ phân phối Chi-Square
            self._stat = d2_sum
            self._df_chi2 = max(1, df_total) # Tránh df <= 0
            self._p_value = 1 - stats.chi2.cdf(self._stat, self._df_chi2)
            self._is_mcar = self._p_value > self._alpha
            
            # Gỡ bỏ cột Pattern tạm thời để không ảnh hưởng dữ liệu gốc
            df.drop(columns=['Pattern'], inplace=True)
            
            self._analyze()
            self._status = "Success"
            
        except Exception as e:
            self._status = f"Failed in test() — {str(e)}"

    def _analyze(self):
        """
        Phân tích kết luận và hiển thị báo cáo chi tiết.
        
        Input:
            Không có.
        
        Output:
            None.
        """
        if self._stat is None:
            display(Markdown("### Dataset không có dữ liệu thiếu."))
            return

        conclusion = "**KẾT LUẬN: Dữ liệu bị thiếu hoàn toàn ngẫu nhiên (MCAR).**" if self._is_mcar else "**KẾT LUẬN: Dữ liệu KHÔNG thiếu ngẫu nhiên (Bác bỏ MCAR). Có thể là MAR hoặc MNAR.**"
        
        explanation = ""
        if self._is_mcar:
            explanation = "Dựa vào kiểm định Little's MCAR (1988), p-value lớn hơn mức ý nghĩa Alpha. Ta chấp nhận giả thuyết H0. Sự thiếu hụt dữ liệu không có tính hệ thống. \n\n👉 **Khuyến nghị:** Bạn có thể an tâm sử dụng các phương pháp đơn giản như điền Mean, Median, Mode hoặc Listwise Deletion (xóa dòng) mà không sợ làm sai lệch phân phối gốc của tập dữ liệu."
        else:
            explanation = "Dựa vào kiểm định Little's MCAR, ta bác bỏ giả thuyết H0 (Dữ liệu thiếu ngẫu nhiên). Sự phân bố của các khoảng trống (Missing patterns) có sự sai lệch đáng kể so với kỳ vọng xác suất. \n\n👉 **Khuyến nghị:** Việc điền khuyết bằng Mean/Median lúc này là **SAI LẦM** vì nó sẽ phá vỡ phương sai của dữ liệu. Cần phải sử dụng Logistic Regression để tìm xem biến nào gây ra sự mất mát này, từ đó áp dụng các thuật toán điền khuyết dựa trên mối quan hệ như **KNN Imputer** hoặc **MICE**."

        md = f"""
## Nhận xét: Kiểm định Little's MCAR (Missing Completely At Random)
*(Sử dụng Kiểm định Toàn cục Chi-Square D²)*

### 1. Kết quả Thống kê
- **Giả thuyết H0:** Sự thiếu dữ liệu là hoàn toàn ngẫu nhiên (MCAR).
- **Giá trị Chi-Square ($\\chi^2$):** `{self._stat:.4f}`
- **Bậc tự do (df):** `{self._df_chi2}`
- **p-Value:** `{self._p_value:.4e}` (Alpha = `{self._alpha}`)

### 2. Kết luận
- {conclusion}

### 3. Diễn giải & Hướng xử lý
- {explanation}
"""
        display(Markdown(md))
