import pandas as pd
from scipy import stats
from core import Service

class ANOVATestAnalysis(Service):
    def __init__(self):
        """
        Khởi tạo đối tượng ANOVATestAnalysis.
        
        Input:
            None.
        
        Output:
            None.
        """
        self._step_name = "Statistic Test: One-Way ANOVA"
        self._status = "Initialized"
        self.anova_results = []
        
    def run(self, df_edge: pd.DataFrame, alpha: float = 0.05):
        """
        Thực hiện kiểm định ANOVA một chiều cho từng phương pháp phát hiện cạnh.
        
        Input:
            df_edge: DataFrame chứa dữ liệu mật độ cạnh.
            alpha: Mức ý nghĩa thống kê (mặc định là 0.05).
        
        Output:
            DataFrame chứa kết quả kiểm định ANOVA cho từng phương pháp.
        """
        print(f"\n[STATISTIC] Bắt đầu kiểm định ANOVA (Mức ý nghĩa alpha = {alpha})...")
        methods = df_edge['Method'].unique()
        
        for method in methods:
            subset = df_edge[df_edge['Method'] == method]
            
            # Gom nhóm dữ liệu Edge Density theo từng lớp (Class)
            groups = []
            for cls in subset['Class'].unique():
                groups.append(subset[subset['Class'] == cls]['Edge_Density'].values)
                
            # Chạy hàm thống kê F-Test One-Way ANOVA của Scipy
            f_stat, p_value = stats.f_oneway(*groups)
            
            is_significant = p_value < alpha
            conclusion = "Có sự khác biệt" if is_significant else "Không có sự khác biệt"
            
            self.anova_results.append({
                "Method": method,
                "F-Statistic": f_stat,
                "P-Value": p_value,
                "Significant": is_significant,
                "Conclusion": conclusion
            })
            
            print(f"   -> {method}: P-Value = {p_value:.2e} | Thống kê F = {f_stat:.2f} | {conclusion}")
            
        self._status = "Success"
        self.log()
        return pd.DataFrame(self.anova_results)

    def visitImageDataset(self, dataset):
        """
        Hàm xử lý cho tập dữ liệu hình ảnh (chưa được triển khai trong lớp này).
        
        Input:
            dataset: Đối tượng dữ liệu đầu vào.
        
        Output:
            None.
        """
        pass 

    def log(self):
        """
        Ghi log trạng thái của tiến trình.
        
        Input:
            None.
        
        Output:
            None.
        """
        print(f"[STATUS] {self._step_name} - {self._status}")
