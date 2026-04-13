import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from core.service_base import Visualization 
from dataset import TableDataset  

class MissingValueMatrix(Visualization):
    def __init__(self, drop_threshold: float = 40.0):
        self._dataset = None
        self._df = None
        self._status = "Not Run"
        self._step_name = "Missing Value Visualization"
        self._drop_threshold = drop_threshold # Ngưỡng bỏ đi (mặc định 40%)
        
        # Biến lưu trữ thống kê
        self._missing_stats = {}

    def run(self):
        if self._dataset is None:
            self._status = "Failed — Dataset is None"
            return
            
        self.visitTableDataset(self._dataset)
        
        if self._status == "Success":
            # Vẽ 2 biểu đồ quan trọng nhất của missingno
            self._plot_visualizations()
            self._analyze()

    def visitTableDataset(self, obj: TableDataset):
        try:
            self._dataset = obj
            self._df = obj.data
            
            if self._df is None or self._df.empty:
                self._status = "Failed — Dữ liệu trống."
                return
                
            self._compute_stats()
            self._status = "Success"
        except Exception as e:
            self._status = f"Failed — {str(e)}"

    def log(self):
        print("=" * 55)
        print(f"Step    : {self._step_name}")
        print(f"Dataset : {self._dataset._folder_path if self._dataset else 'None'}")
        print(f"Status  : {self._status}")
        print("=" * 55)

    def _compute_stats(self):
        total_rows = len(self._df)
        missing_count = self._df.isnull().sum()
        missing_pct = (missing_count / total_rows) * 100
        
        # 1. Thống kê theo cột
        self._missing_stats['all'] = missing_pct[missing_pct > 0].sort_values(ascending=False)
        self._missing_stats['severe'] = missing_pct[missing_pct > self._drop_threshold]
        self._missing_stats['low'] = missing_pct[(missing_pct > 0) & (missing_pct <= self._drop_threshold)]
        
        # 2. Thống kê theo dòng (Dòng nào thiếu nhiều biến nhất)
        row_missing = self._df.isnull().sum(axis=1)
        self._missing_stats['max_missing_in_row'] = row_missing.max()
        self._missing_stats['rows_with_heavy_missing'] = len(row_missing[row_missing > (self._df.shape[1] * 0.5)])

        # 3. Tương quan thiếu (Correlation of missingness)
        # Tạo dataframe boolean (True nếu missing, False nếu không)
        missing_df = self._df.isnull().astype(int)
        # Chỉ tính tương quan giữa các cột có dữ liệu thiếu
        cols_with_missing = missing_pct[missing_pct > 0].index
        if len(cols_with_missing) > 1:
            self._missing_stats['corr'] = missing_df[cols_with_missing].corr()
        else:
            self._missing_stats['corr'] = None

    def _plot_visualizations(self):
        # Thiết lập để 2 biểu đồ hiển thị liên tiếp
        fig = plt.figure(figsize=(15, 6))
        
        # 1. Ma trận thiếu (Matrix)
        ax1 = fig.add_subplot(121)
        msno.matrix(self._df, ax=ax1, sparkline=False, fontsize=10, color=(0.2, 0.4, 0.6))
        ax1.set_title("Ma trận Dữ liệu thiếu (Trắng là thiếu)", fontsize=14, pad=20)
        
        # 2. Heatmap tương quan thiếu
        ax2 = fig.add_subplot(122)
        if self._missing_stats['corr'] is not None:
            msno.heatmap(self._df, ax=ax2, fontsize=10, cmap="coolwarm")
            ax2.set_title("Tương quan Thiếu (Missingness Correlation)", fontsize=14, pad=20)
        else:
            ax2.text(0.5, 0.5, "Không đủ cột thiếu để vẽ tương quan", ha='center', va='center')
            
        plt.tight_layout()
        plt.show()

    def _analyze(self):
        severe_cols = ", ".join([f"`{c}` ({v:.1f}%)" for c, v in self._missing_stats['severe'].items()])
        low_cols = ", ".join([f"`{c}` ({v:.1f}%)" for c, v in self._missing_stats['low'].items()][:5]) + "..."
        
        # Phân tích tương quan
        corr_analysis = "Không phát hiện tương quan đặc biệt."
        if self._missing_stats['corr'] is not None:
            # Tìm các cặp có tương quan thiếu >= 0.7
            corr_mat = self._missing_stats['corr']
            upper_tri = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
            high_corr = upper_tri.unstack().dropna()
            high_corr = high_corr[high_corr >= 0.7]
            
            if not high_corr.empty:
                pairs = "\n".join([f"  - `{idx[0]}` & `{idx[1]}` (r = {val:.2f})" for idx, val in high_corr.items()])
                corr_analysis = f"Phát hiện các cặp biến **rủ nhau cùng thiếu**:\n{pairs}\n  > *Ý nghĩa:* Quá trình thu thập có tính hệ thống (ví dụ: Nếu không có giấy phép xây dựng thì cũng không có ngày bắt đầu thi công)."

        md = f"""
## Nhận xét: Trực quan hóa Dữ liệu Thiếu (Missing Values)

### 1. Phân bố Dữ liệu theo Cột
- **Cột cần loại bỏ (Thiếu > {self._drop_threshold}%):** {severe_cols if severe_cols else '*Không có*'}
  > Các cột này thiếu quá nhiều thông tin, nếu fill (điền) sẽ sinh ra nhiễu lớn làm sai lệch mô hình. Khuyến nghị `drop`.
- **Cột thiếu ít/vừa (Có thể fill):** {low_cols}

### 2. Phân bố Dữ liệu theo Dòng
- Có dòng bị thiếu tối đa **{self._missing_stats['max_missing_in_row']}** biến cùng lúc.
- Số lượng dòng bị thiếu quá nửa số biến (>50% số cột): **{self._missing_stats['rows_with_heavy_missing']}** dòng.
  > *Nhận xét:* Các dòng thiếu quá nhiều biến khả năng cao là dữ liệu rác hoặc hồ sơ bị hủy giữa chừng. Nên xóa các dòng này trước khi Train model. Nhìn vào biểu đồ Matrix (bên trái), nếu các dải màu trắng vạch ngang qua toàn bộ, chứng tỏ missing có xu hướng "tụm lại" ở các hồ sơ hỏng.

### 3. Tương quan Dữ liệu Thiếu (Missingness Correlation)
{corr_analysis}
"""
        display(Markdown(md))
