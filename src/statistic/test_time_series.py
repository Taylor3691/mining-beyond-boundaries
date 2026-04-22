from __future__ import annotations
from typing import Any
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from core import Testing
from visualization.relationship import plot_granger_causality_directed_graph

class GrangerCausalityTesting(Testing):
	"""
	Kiểm định Granger causality cho dữ liệu đa biến (multivariate time series).

	Ý nghĩa:
	- Nếu p-value < alpha cho cặp (X -> Y), ta kết luận X "Granger-cause" Y.
	- Lưu ý: đây là quan hệ nhân quả theo nghĩa dự báo (predictive causality),
	  không phải nhân quả tuyệt đối theo cơ chế tự nhiên.
	"""

	def __init__(
		self,
		variables: list[str] | None = None,
		max_lag: int = 5,
		alpha: float = 0.05,
		test_stat: str = "ssr_chi2test",
		verbose: bool = False,
	):
		"""
		Khởi tạo bộ kiểm định Granger Causality.

		Input:
			variables: Danh sách các biến cần kiểm định.
			max_lag: Độ trễ tối đa.
			alpha: Mức ý nghĩa thống kê.
			test_stat: Loại thống kê kiểm định (mặc định ssr_chi2test).
		"""
		super().__init__()
		self.variables = variables
		self.max_lag = max_lag
		self.alpha = alpha
		self.test_stat = test_stat
		self.verbose = verbose
		self.step_name = "Granger Causality Test"
		self.dataset_name = "Unknown"
		self.status = "Pending"
		self.p_value_matrix: pd.DataFrame | None = None
		self.best_lag_matrix: pd.DataFrame | None = None
		self.binary_matrix: pd.DataFrame | None = None

	def log(self):
		"""In thông tin tóm tắt kết quả kiểm định."""
		print("=" * 70)
		print(f"Step       : {self.step_name}")
		print(f"Dataset    : {self.dataset_name}")
		print(f"Status     : {self.status}")
		print(f"Max Lag    : {self.max_lag}")
		print(f"Alpha      : {self.alpha}")
		print(f"Test Stat  : {self.test_stat}")

		if self.binary_matrix is not None:
			edge_count = int(self.binary_matrix.values.sum() - np.trace(self.binary_matrix.values))
			print(f"Significant edges (X -> Y): {edge_count}")
		print("=" * 70)

	def visitImageDataset(self, obj=None):
		"""Kiểm định không hỗ trợ dữ liệu hình ảnh."""
		raise ValueError("GrangerCausalityTesting chỉ hỗ trợ dữ liệu dạng bảng/time series.")

	def visitTableDataset(self, obj):
		"""Thực thi kiểm định trên đối tượng TableDataset."""
		self.dataset_name = getattr(obj, "_folder_path", type(obj).__name__)
		if not hasattr(obj, "data") or obj.data is None:
			raise ValueError("Dataset không hợp lệ hoặc chưa nạp dữ liệu (obj.data is None).")

		if not isinstance(obj.data, pd.DataFrame):
			raise TypeError("obj.data phải là pandas.DataFrame.")

		self.test(obj.data)

	def run(self, obj):
		"""Điểm vào thực thi kiểm định Granger."""
		try:
			obj_type = type(obj).__name__
			if obj_type in {"TableDataset", "TimeSeriesDataset"}:
				self.visitTableDataset(obj)
			else:
				if hasattr(obj, "data") and isinstance(getattr(obj, "data"), pd.DataFrame):
					self.visitTableDataset(obj)
				else:
					raise ValueError(f"Lớp {self.__class__.__name__} chưa hỗ trợ type: {obj_type}")
			self.status = "Success"
		except Exception as exc:
			self.status = f"Failed ({exc})"
			raise
		finally:
			self.log()

	def test(self, data: pd.DataFrame):
		"""
		Chạy Granger causality test cho mọi cặp biến số.

		Input:
			data: DataFrame chứa các biến số theo trục thời gian.

		Output:
			Bộ ba ma trận (p_value_matrix, best_lag_matrix, binary_matrix).
		"""
		# 1) Chỉ giữ cột số để đảm bảo statsmodels xử lý được
		numeric_df = data.select_dtypes(include=[np.number]).copy()

		if numeric_df.shape[1] < 2:
			raise ValueError("Cần ít nhất 2 cột số để chạy Granger causality test.")

		# 2) Chọn biến theo cấu hình (nếu có), ngược lại dùng toàn bộ cột số
		if self.variables is not None:
			missing_vars = [col for col in self.variables if col not in numeric_df.columns]
			if missing_vars:
				raise ValueError(f"Không tìm thấy các biến trong dữ liệu: {missing_vars}")
			selected_cols = list(self.variables)
		else:
			selected_cols = list(numeric_df.columns)

		work_df = numeric_df[selected_cols].copy()

		# 3) Khởi tạo ma trận kết quả
		p_value_matrix = pd.DataFrame(np.nan, index=selected_cols, columns=selected_cols)
		best_lag_matrix = pd.DataFrame(np.nan, index=selected_cols, columns=selected_cols)

		# 4) Duyệt tất cả cặp (cause -> effect)
		for cause in selected_cols:
			for effect in selected_cols:
				if cause == effect:
					# Đường chéo chính không xét nhân quả
					p_value_matrix.loc[cause, effect] = np.nan
					best_lag_matrix.loc[cause, effect] = np.nan
					continue

				# Granger yêu cầu 2 chuỗi cùng thời gian và không có NaN
				# Thứ tự cột phải là [effect, cause] theo API statsmodels
				pair_df = work_df[[effect, cause]].dropna()

				# Cần đủ dữ liệu cho max_lag; nếu không đủ thì bỏ qua cặp này
				if len(pair_df) <= (self.max_lag + 1):
					continue

				try:
					# statsmodels hiện phát FutureWarning cho tham số verbose,
					# nên ta chặn warning này để output sạch hơn.
					with warnings.catch_warnings():
						warnings.filterwarnings(
							"ignore",
							message="verbose is deprecated since functions should not print results",
							category=FutureWarning,
						)
						test_result = grangercausalitytests(
							pair_df,
							maxlag=self.max_lag,
							verbose=self.verbose,
						)

					# Lấy p-value theo từng lag từ test_stat đã chọn
					lag_to_p = {}
					for lag in range(1, self.max_lag + 1):
						lag_to_p[lag] = float(test_result[lag][0][self.test_stat][1])

					# Chọn lag cho p-value nhỏ nhất để biểu diễn quan hệ mạnh nhất
					best_lag = min(lag_to_p, key=lag_to_p.get)
					best_p = lag_to_p[best_lag]

					p_value_matrix.loc[cause, effect] = best_p
					best_lag_matrix.loc[cause, effect] = int(best_lag)
				except Exception:
					continue

		# 5) Chuyển sang ma trận nhị phân theo ngưỡng alpha
		binary_matrix = (p_value_matrix < self.alpha).astype(int)
		for col in selected_cols:
			binary_matrix.loc[col, col] = 0

		self.p_value_matrix = p_value_matrix
		self.best_lag_matrix = best_lag_matrix
		self.binary_matrix = binary_matrix

		return self.p_value_matrix, self.best_lag_matrix, self.binary_matrix

	def visualize_directed_graph(self, title: str = "Granger Causality Directed Graph", save_path: str = None, show_edge_labels: bool = True):
		"""Trực quan hóa kết quả thành đồ thị hướng."""
		if self.p_value_matrix is None:
			raise ValueError("Chưa có kết quả kiểm định. Hãy chạy run(obj) hoặc test(data) trước.")

		plot_granger_causality_directed_graph(
			p_value_matrix=self.p_value_matrix,
			alpha=self.alpha,
			title=title,
			save_path=save_path,
			show_edge_labels=show_edge_labels,
		)

	def get_significant_pairs(self) -> list[dict[str, Any]]:
		"""Trả về danh sách các cặp biến có ý nghĩa thống kê."""
		if self.p_value_matrix is None or self.best_lag_matrix is None:
			raise ValueError("Chưa có kết quả kiểm định. Hãy chạy run(obj) hoặc test(data) trước.")

		results: list[dict[str, Any]] = []
		for cause in self.p_value_matrix.index:
			for effect in self.p_value_matrix.columns:
				if cause == effect:
					continue

				p_value = self.p_value_matrix.loc[cause, effect]
				best_lag = self.best_lag_matrix.loc[cause, effect]

				if pd.notna(p_value) and p_value < self.alpha:
					results.append(
						{
							"cause": cause,
							"effect": effect,
							"p_value": float(p_value),
							"best_lag": int(best_lag) if pd.notna(best_lag) else None,
						}
					)

		# Sắp xếp theo p-value tăng dần để xem quan hệ mạnh trước
		results.sort(key=lambda item: item["p_value"])
		return results
