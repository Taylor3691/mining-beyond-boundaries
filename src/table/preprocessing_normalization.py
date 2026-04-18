"""Normalization preprocessors for tabular data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler

from core.service_base import Preprocessing


class _BaseTableNormalization(Preprocessing):
	"""Shared workflow for table normalization services."""

	def __init__(self, step_name: str):
		self._step_name = step_name
		self._dataset_path = "Unknown"
		self._status = "Initialized"
		self._error_message = ""

		self._scaler = None
		self._numeric_columns: list[str] = []
		self._transformed: pd.DataFrame | np.ndarray | None = None

	@property
	def numeric_columns(self) -> list[str]:
		return self._numeric_columns

	@property
	def transformed_features(self) -> pd.DataFrame | np.ndarray | None:
		return self._transformed

	def _to_dataframe(self, X: Any) -> tuple[pd.DataFrame, bool]:
		if isinstance(X, pd.DataFrame):
			return X.copy(), True

		X_arr = np.asarray(X)
		if X_arr.ndim != 2:
			raise ValueError("Input phải là mảng 2 chiều hoặc pandas DataFrame.")

		columns = [f"feature_{i}" for i in range(X_arr.shape[1])]
		return pd.DataFrame(X_arr, columns=columns), False

	def _build_scaler(self, n_samples: int):
		raise NotImplementedError("Subclasses must implement _build_scaler.")

	def fit(self, X, y=None):
		df, _ = self._to_dataframe(X)
		self._numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

		if not self._numeric_columns:
			raise ValueError("Không tìm thấy cột số để chuẩn hóa.")

		self._scaler = self._build_scaler(n_samples=len(df))
		self._scaler.fit(df[self._numeric_columns])
		return self

	def transform(self, X):
		if self._scaler is None:
			raise ValueError("Model chưa fit. Hãy gọi fit trước.")

		df, is_dataframe = self._to_dataframe(X)
		missing_cols = [col for col in self._numeric_columns if col not in df.columns]
		if missing_cols:
			raise ValueError(f"Thiếu cột khi transform: {missing_cols}")

		transformed_df = df.copy()
		scaled_values = self._scaler.transform(df[self._numeric_columns])
		scaled_df = pd.DataFrame(scaled_values, columns=self._numeric_columns, index=transformed_df.index)
		# Assign theo block thay vì .loc inplace để pandas tự nâng kiểu dữ liệu an toàn.
		transformed_df[self._numeric_columns] = scaled_df

		self._transformed = transformed_df if is_dataframe else transformed_df.to_numpy()
		return self._transformed

	def fit_transform(self, X, y=None):
		self.fit(X, y)
		return self.transform(X)

	def visitImageDataset(self, obj):
		print(f"[WARNING] {self.__class__.__name__} không hỗ trợ ImageDataset.")
		return

	def visitTableDataset(self, obj):
		self._dataset_path = getattr(obj, "_folder_path", "Unknown")

		try:
			X = getattr(obj, "features", None)
			y = getattr(obj, "target", None)
			if X is None:
				raise ValueError(
					"Dataset chưa có features. Hãy gọi set_target(target_column) trước khi preprocessing."
				)

			X_scaled = self.fit_transform(X)
			if not isinstance(X_scaled, pd.DataFrame):
				columns = X.columns if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X_scaled.shape[1])]
				X_scaled = pd.DataFrame(X_scaled, columns=columns, index=getattr(X, "index", None))

			obj._features = X_scaled

			target_col = getattr(obj, "_target_column", None)
			if target_col is not None and y is not None:
				y_series = y if isinstance(y, pd.Series) else pd.Series(y, name=target_col, index=X_scaled.index)
				obj.data = pd.concat([X_scaled, y_series.rename(target_col)], axis=1)
				obj.set_target(target_col)
			else:
				obj.data = X_scaled

			self._status = "Success"
		except Exception as exc:
			self._status = "Failed"
			self._error_message = str(exc)

	def run(self, obj):
		self.visitTableDataset(obj)

	def log(self):
		print("\n" + "=" * 55)
		print(f"Step      : {self._step_name}")
		print(f"Dataset   : {self._dataset_path}")
		print(f"Status    : {self._status}")
		if self._status == "Success":
			print(f"Columns   : {len(self._numeric_columns)} numeric column(s)")
			print(f"Names     : {self._numeric_columns}")
		else:
			print(f"Error     : {self._error_message}")
		print("=" * 55)


class StatisticalNormalization(_BaseTableNormalization):
	"""
	One statistical normalization class for:
	- Min-Max scaling
	- Z-score scaling
	- Robust scaling
	"""

	SUPPORTED_METHODS = {"minmax", "zscore", "robust"}

	def __init__(
		self,
		method: str = "minmax",
		feature_range: tuple[float, float] = (0.0, 1.0),
		with_mean: bool = True,
		with_std: bool = True,
		with_centering: bool = True,
		with_scaling: bool = True,
		quantile_range: tuple[float, float] = (25.0, 75.0),
	):
		method = method.lower()
		if method not in self.SUPPORTED_METHODS:
			raise ValueError(
				f"method phải thuộc {sorted(self.SUPPORTED_METHODS)}, nhận được: {method}"
			)

		super().__init__(step_name=f"Statistical Normalization ({method})")
		self._method = method

		self._feature_range = feature_range
		self._with_mean = with_mean
		self._with_std = with_std
		self._with_centering = with_centering
		self._with_scaling = with_scaling
		self._quantile_range = quantile_range

	def _build_scaler(self, n_samples: int):
		if self._method == "minmax":
			return MinMaxScaler(feature_range=self._feature_range)
		if self._method == "zscore":
			return StandardScaler(with_mean=self._with_mean, with_std=self._with_std)
		return RobustScaler(
			with_centering=self._with_centering,
			with_scaling=self._with_scaling,
			quantile_range=self._quantile_range,
		)


class QuantileNormalization(_BaseTableNormalization):
	"""Quantile transform normalization with uniform or normal output."""

	SUPPORTED_OUTPUTS = {"uniform", "normal"}

	def __init__(
		self,
		output_distribution: str = "uniform",
		n_quantiles: int = 1000,
		subsample: int = 100_000,
		random_state: int = 42,
	):
		output_distribution = output_distribution.lower()
		if output_distribution not in self.SUPPORTED_OUTPUTS:
			raise ValueError(
				"output_distribution phải là 'uniform' hoặc 'normal'."
			)
		if n_quantiles <= 0:
			raise ValueError("n_quantiles phải > 0.")

		super().__init__(
			step_name=f"Quantile Normalization (output={output_distribution})"
		)

		self._output_distribution = output_distribution
		self._n_quantiles = n_quantiles
		self._subsample = subsample
		self._random_state = random_state

	def _build_scaler(self, n_samples: int):
		effective_quantiles = min(self._n_quantiles, max(1, n_samples))
		return QuantileTransformer(
			n_quantiles=effective_quantiles,
			output_distribution=self._output_distribution,
			subsample=self._subsample,
			random_state=self._random_state,
			copy=True,
		)

