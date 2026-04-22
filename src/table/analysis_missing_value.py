import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from core.service_base import Visualization 
from table.dataset import TableDataset  

class MissingValueMatrix(Visualization):
    """Trực quan hóa và phân tích dữ liệu thiếu (Missing Values) trong tập dữ liệu bảng."""

    def __init__(self, drop_threshold: float = 40.0):
        """
        Khởi tạo bộ phân tích dữ liệu thiếu.

        Input:
            drop_threshold: Ngưỡng % thiếu để khuyến nghị loại bỏ cột (mặc định 40%).

        Output:
            None.
        """
        self._dataset = None
        self._df = None
        self._status = "Not Run"
        self._step_name = "Missing Value Visualization"
        self._drop_threshold = drop_threshold # Ngưỡng bỏ đi (mặc định 40%)
        
        # Biến lưu trữ thống kê
        self._missing_stats = {}

    def run(self):
        """
        Thực thi quy trình phân tích dữ liệu thiếu: thống kê, trực quan và nhận xét.

        Input:
            Không có (sử dụng _dataset đã gán).

        Output:
            None.
        """
        if self._dataset is None:
            self._status = "Failed — Dataset is None"
            return
            
        self.visitTableDataset(self._dataset)
        
        if self._status == "Success":
            # Vẽ 2 biểu đồ quan trọng nhất của missingno
            self._plot_visualizations()
            self._analyze()

    def visitTableDataset(self, obj: TableDataset):
        """
        Gắn TableDataset và tính toán thống kê dữ liệu thiếu.

        Input:
            obj: Đối tượng TableDataset chứa dữ liệu cần phân tích.

        Output:
            None (cập nhật _status và _missing_stats nội bộ).
        """
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
    def visitImageDataset(self, obj):
        """
        Không hỗ trợ dữ liệu hình ảnh.

        Input:
            obj: Đối tượng ImageDataset.

        Output:
            None.
        """
        self._status = "Failed — MissingValueMatrix không hỗ trợ tập dữ liệu Ảnh (ImageDataset)."

    def visitTextDataset(self, obj):
        """
        Không hỗ trợ dữ liệu văn bản.

        Input:
            obj: Đối tượng TextDataset.

        Output:
            None.
        """
        self._status = "Failed — MissingValueMatrix không hỗ trợ tập dữ liệu Văn bản (TextDataset)."

    def visitTimeSeriesDataset(self, obj):
        """
        Không hỗ trợ dữ liệu chuỗi thời gian.

        Input:
            obj: Đối tượng TimeSeriesDataset.

        Output:
            None.
        """
        self._status = "Failed — MissingValueMatrix không hỗ trợ tập dữ liệu Chuỗi thời gian (TimeSeriesDataset)."
    def log(self):
        """
        In trạng thái thực thi của bước phân tích.

        Input:
            Không có.

        Output:
            None (in ra màn hình).
        """
        print("=" * 55)
        print(f"Step    : {self._step_name}")
        print(f"Dataset : {self._dataset._folder_path if self._dataset else 'None'}")
        print(f"Status  : {self._status}")
        print("=" * 55)

    def _compute_stats(self):
        """
        Tính toán thống kê dữ liệu thiếu theo cột, dòng và tương quan thiếu.

        Input:
            Không có (sử dụng _df nội bộ).

        Output:
            None (cập nhật _missing_stats dict).
        """
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
        """
        Vẽ ma trận dữ liệu thiếu (Matrix) và Heatmap tương quan thiếu.

        Input:
            Không có (sử dụng _df và _missing_stats nội bộ).

        Output:
            None (hiển thị biểu đồ matplotlib).
        """
        # 1. TỐI ƯU CỘT: Vẫn nên giữ việc lọc các cột CÓ DỮ LIỆU THIẾU để trục X bớt chật
        missing_cols = self._df.columns[self._df.isnull().any()].tolist()
        df_plot = self._df[missing_cols] if len(missing_cols) > 0 else self._df
        
        # 2. KÉO DÃN KHUNG HÌNH: Tăng size lên 30x12 (Cực rộng)
        fig = plt.figure(figsize=(30, 12))
        
        # 3. Ma trận thiếu (Matrix) - CHẠY FULL DÒNG
        ax1 = fig.add_subplot(121)
        # sparkline=False để bỏ cái đường loằng ngoằng bên phải, dành diện tích cho chữ
        msno.matrix(df_plot, ax=ax1, sparkline=False, fontsize=10, color=(0.2, 0.4, 0.6))
        ax1.set_title(f"Ma trận Dữ liệu thiếu (Toàn bộ {len(self._df)} dòng)", fontsize=20, pad=40)
        
        # 4. Heatmap tương quan thiếu
        ax2 = fig.add_subplot(122)
        if self._missing_stats['corr'] is not None:
            msno.heatmap(self._df, ax=ax2, fontsize=10, cmap="coolwarm")
            ax2.set_title("Tương quan Thiếu (Missingness Correlation)", fontsize=20, pad=40)
        else:
            ax2.text(0.5, 0.5, "Không đủ cột thiếu để vẽ tương quan", ha='center', va='center')
            
        # 5. CÁCH LÝ CHỮ: Tăng wspace=0.5 để đẩy 2 hình xa nhau ra
        plt.subplots_adjust(wspace=0.5, bottom=0.2) 
        plt.show()

    def _analyze(self):
        """
        Phân tích kết quả và hiển thị nhận xét chi tiết dưới dạng Markdown.

        Input:
            Không có.

        Output:
            None (hiển thị Markdown trong IPython).
        """
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
