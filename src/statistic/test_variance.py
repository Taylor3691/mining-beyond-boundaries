from core import Testing
from image import ImageDataset
import numpy as np
from scipy import stats
from table.dataset import TableDataset

class LeveneTesting(Testing):
	"""Kiểm định Levene về tính đồng nhất của phương sai giữa các mẫu."""

	def __init__(self, alpha: float = 0.05, center: str = "median"):
		"""
		Khởi tạo kiểm định Levene.

		Input:
			alpha: Mức ý nghĩa thống kê.
			center: Phương pháp tính tâm (mean, median, trimmed).
		"""
		if center not in {"mean", "median", "trimmed"}:
			raise ValueError("center phải là một trong: 'mean', 'median', 'trimmed'")

		self.step_name = "Variance Homogeneity Check"
		self.test_name = "Levene's Test"
		self.alpha = alpha
		self.center = center
		self.h0_hypothesis = "Hai tập dữ liệu (Gốc và Đã xử lý) có phương sai bằng nhau."

		self.statistic = None
		self.p_value = None
		self.conclusion = ""
		self.is_rejected = False

	def visitImageDataset(self, obj: ImageDataset):
		"""Kiểm định phương sai cho dữ liệu hình ảnh."""
		if not hasattr(obj, "_origin_images") or not hasattr(obj, "_processed_images"):
			raise AttributeError("ImageDataset phải có thuộc tính '_origin_images' và '_processed_images'")

		try:
			data_orig_flat = np.concatenate(obj._origin_images).ravel()
			data_proc_flat = np.concatenate(obj._processed_images).ravel()
		except Exception as e:
			raise ValueError(f"Lỗi khi xử lý mảng ảnh: {e}")
		self.test(data_orig_flat, data_proc_flat)
		return

	def visitTableDataset(self, obj: TableDataset):
		"""Kiểm định phương sai cho dữ liệu bảng."""
		if not hasattr(obj, "_origin_data") or getattr(obj, "_origin_data") is None:
			raise AttributeError("TableDataset phải có thuộc tính '_origin_data' để thực hiện Levene Test.")

		try:
			orig_numeric = obj._origin_data.select_dtypes(include=[np.number]).values.ravel()
			proc_numeric = obj.data.select_dtypes(include=[np.number]).values.ravel()
			data_orig_flat = orig_numeric[np.isfinite(orig_numeric)]
			data_proc_flat = proc_numeric[np.isfinite(proc_numeric)]
		except Exception as e:
			raise ValueError(f"Lỗi khi xử lý dữ liệu bảng: {e}")
		self.test(data_orig_flat, data_proc_flat)
		return

	def run(self, obj):
		"""Thực thi kiểm định dựa trên kiểu đối tượng truyền vào."""
		obj_type = type(obj).__name__
		if obj_type == "ImageDataset":
			self.visitImageDataset(obj)
		elif obj_type == "TableDataset":
			self.visitTableDataset(obj)
		else:
			raise ValueError(f"Lớp {self.__class__.__name__} chưa hỗ trợ type: {obj_type}")
		return

	def log(self):
		"""In tóm tắt kết quả kiểm định ra màn hình."""
		print(f"Step: {self.step_name}")
		print(f"Kiểm định: {self.test_name}")
		print(f"Giả thuyết H0: {self.h0_hypothesis}")
		if self.statistic is None or self.p_value is None:
			print("Kết quả: Chưa có. Hãy chạy run(obj) trước.")
			return
		print(f"Center: {self.center}")
		print(f"P-value: {self.p_value:.10f}")
		print(f"Statistic: {self.statistic:.6f}")

		if self.is_rejected:
			print(f"Kết luận: {self.conclusion}.")
			print(f"Thống kê cho thấy phương sai đã thay đổi có ý nghĩa (p <= {self.alpha}).")
		else:
			print(f"Kết luận: {self.conclusion}.")
			print("Chưa có bằng chứng đủ mạnh để kết luận phương sai thay đổi.")
		return

	def test(self, data_orig: np.ndarray, data_proc: np.ndarray):
		"""
		Thực hiện tính toán thống kê Levene.

		Input:
			data_orig: Mảng dữ liệu gốc.
			data_proc: Mảng dữ liệu đã xử lý.

		Output:
			Bộ tham số (statistic, p_value).
		"""
		data_orig = np.asarray(data_orig, dtype=float)
		data_proc = np.asarray(data_proc, dtype=float)
		data_orig = data_orig[np.isfinite(data_orig)]
		data_proc = data_proc[np.isfinite(data_proc)]
		if data_orig.size < 2 or data_proc.size < 2:
			raise ValueError("Cần tối thiểu 2 giá trị hợp lệ cho mỗi nhóm để chạy Levene's test")

		self.statistic, self.p_value = stats.levene(data_orig, data_proc, center=self.center)
		self.is_rejected = self.p_value <= self.alpha
		if self.is_rejected:
			self.conclusion = "Bác bỏ giả thuyết H0"
		else:
			self.conclusion = "Chấp nhận giả thuyết H0"

		return self.statistic, self.p_value
