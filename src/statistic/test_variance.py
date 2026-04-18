from core import Testing
from image import ImageDataset
import numpy as np
from scipy import stats
from table.dataset import TableDataset


class LeveneTesting(Testing):
	"""
	Levene's test for checking homogeneity of variances between
	original and processed datasets.

	H0: Two groups have equal variances.
	H1: Two groups do not have equal variances.
	"""

	def __init__(self, alpha: float = 0.05, center: str = "median"):
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
		obj_type = type(obj).__name__
		if obj_type == "ImageDataset":
			self.visitImageDataset(obj)
		elif obj_type == "TableDataset":
			self.visitTableDataset(obj)
		else:
			raise ValueError(f"Lớp {self.__class__.__name__} chưa hỗ trợ type: {obj_type}")
		return

	def log(self):
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
