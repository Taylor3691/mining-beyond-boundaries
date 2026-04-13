import pandas as pd
import numpy as np
from typing import Any, Dict

# Kế thừa từ Visualization
from core import Visualization 
from visualization.relationship import plot_pearson_heatmap, plot_spearman_heatmap

class CorrelationAnalysis(Visualization):
    def __init__(self, method: str = 'pearson', threshold: float = 0.7):
        method = method.lower()
        if method not in ['pearson', 'spearman']:
            raise ValueError("Method not support. Please choose 'pearson' or 'spearman'.")
        
        self._method = method
        self._threshold = threshold
        self._corr_matrix = None
        self._stats: Dict[str, Any] = {}
        self._step_name = f"Correlation Analysis ({method.capitalize()})"
        self._dataset_name = "Tabular Dataset"
        self._status = "Initialized"
        return

    # ====================================================================
    # 1. CÁC HÀM ABSTRACT BẮT BUỘC (Thỏa mãn tính Đa hình của OOP)
    # ====================================================================
    def run(self, obj):
        """Hàm thực thi chính"""
        if isinstance(obj, pd.DataFrame):
            self.visitDataFrame(obj)
        else:
            # Nếu truyền vào là object Dataset, cố gắng lấy thuộc tính chứa Data
            df = getattr(obj, 'data', obj)
            self.visitDataFrame(df)
        return

    def visitImageDataset(self, obj):
        """
        [FIX BUGS]: Hàm này BẮT BUỘC phải có mặt để thỏa mãn Abstract Class của lõi Core.
        Do class này chuyên xử lý Bảng (Tabular) nên ta sẽ bỏ qua (pass) nếu lỡ truyền ảnh vào.
        """
        print(f"[WARNING] Class {self.__class__.__name__} không hỗ trợ xử lý ImageDataset.")
        return

    # ====================================================================
    # 2. LOGIC XỬ LÝ CHÍNH
    # ====================================================================
    def visitDataFrame(self, obj):
        try:
            print(f"   [INFO] Đang thực thi {self._step_name}...")
            
            # Lấy dữ liệu 
            df = obj if isinstance(obj, pd.DataFrame) else getattr(obj, 'data', obj)
            
            # Chỉ lấy các cột dạng số (Int, Float)
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                raise ValueError("Không có cột dạng số nào trong tập dữ liệu!")

            # 1. Tính toán ma trận tương quan
            self._corr_matrix = numeric_df.corr(method=self._method)

            # 2. Visualize heatmap tương ứng
            if self._method == 'pearson':
                plot_pearson_heatmap(numeric_df, title="Pearson Correlation Heatmap")
            elif self._method == 'spearman':
                plot_spearman_heatmap(numeric_df, title="Spearman Correlation Heatmap")

            # 3. Lọc các biến có tương quan cao/thấp (Lưu vào stats để in ra log)
            self._extract_high_low_correlation()
            
            self._status = "Success"
        except Exception as e:
            self._status = f"Failed ({str(e)})"
            print(self._status)
        finally:
            self.log()
        return

    def log(self):
        print(f"\nMethod: {self._method.capitalize()}")
        print(f"Status: {self._status}")
        
        # In ra các cặp tương quan mạnh và yếu đã được lưu trong stats
        for key, value in self._stats.items():
            print(f"  - {key}:")
            if isinstance(value, list) and len(value) > 0:
                for item in value:
                    print(f"      + {item}")
            elif isinstance(value, list) and len(value) == 0:
                print("      + (Không có cặp nào)")
            else:
                print(f"      + {value}")
        return

    def _extract_high_low_correlation(self):
        """Hàm nội bộ: Lọc ra các biến có tương quan cao và cực thấp dựa vào ngưỡng T"""
        if self._corr_matrix is None:
            return

        corr_pairs = self._corr_matrix.unstack().reset_index()
        corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
        
        # Loại bỏ các cặp tự tương quan
        corr_pairs = corr_pairs[corr_pairs['Feature 1'] != corr_pairs['Feature 2']]
        
        # Loại bỏ các cặp trùng lặp (A-B và B-A)
        corr_pairs['pair'] = corr_pairs.apply(
            lambda row: '-'.join(sorted([str(row['Feature 1']), str(row['Feature 2'])])), axis=1
        )
        corr_pairs = corr_pairs.drop_duplicates(subset=['pair']).drop(columns=['pair'])
        
        corr_pairs['Abs_Corr'] = corr_pairs['Correlation'].abs()
        corr_pairs = corr_pairs.sort_values(by='Abs_Corr', ascending=False)

        # Lọc tương quan MẠNH
        high_corr = corr_pairs[corr_pairs['Abs_Corr'] >= self._threshold]
        high_list = [f"[{row['Feature 1']}] & [{row['Feature 2']}]: r = {row['Correlation']:.4f}" 
                     for _, row in high_corr.iterrows()]
        
        # Lọc tương quan CỰC YẾU
        low_corr = corr_pairs.sort_values(by='Abs_Corr', ascending=True).head(5)
        low_list = [f"[{row['Feature 1']}] & [{row['Feature 2']}]: r = {row['Correlation']:.4f}" 
                    for _, row in low_corr.iterrows()]

        self._stats[f'Tương quan MẠNH (|r| >= {self._threshold})'] = high_list
        self._stats['Tương quan CỰC YẾU (Top 5)'] = low_list