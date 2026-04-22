"""Xử lý ma trận đặc trưng cho bài toán dự báo chuỗi thời gian (One-step-ahead forecasting)."""

from __future__ import annotations
from typing import Any, Iterable
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import pacf
from core.service_base import Preprocessing


class FeatureMatrixPredictionPreprocessing(Preprocessing):
	"""Build a full forecasting feature matrix and evaluate a Random Forest model.

	Required feature groups:
	1) PACF-driven lag features.
	2) Multi-window rolling statistics.
	3) Time-derived calendar features.
	4) Decomposition features (trend, seasonal, residual).
	"""

	def __init__(
		self,
		target_column: str = "Confirmed",
		time_column: str = "Date",
		entity_column: str | None = "Country/Region",
		entity_value: str = "World",
		aggregation: str = "sum",
		forecast_horizon: int = 1,
		train_ratio: float = 0.8,
		max_pacf_lags: int = 40,
		pacf_method: str = "ywm",
		rolling_windows: Iterable[int] = (3, 7, 14, 30),
		decomposition_period: int = 7,
		random_state: int = 42,
		n_estimators: int = 300,
		max_depth: int | None = None,
		min_samples_leaf: int = 1,
		n_jobs: int = -1,
		model_kwargs: dict[str, Any] | None = None,
	):
		"""
		Khởi tạo bộ tiền xử lý tạo ma trận đặc trưng cho dự báo chuỗi thời gian một bước.

		Input:
			target_column: Tên cột mục tiêu cần dự báo.
			time_column: Tên cột thời gian dùng để sắp xếp chuỗi.
			entity_column: Tên cột định danh thực thể (quốc gia, vùng...) nếu có.
			entity_value: Giá trị thực thể cần lọc trước khi tổng hợp.
			aggregation: Hàm tổng hợp khi gộp theo thời gian (sum, mean...).
			forecast_horizon: Số bước dự báo về tương lai (mặc định 1).
			train_ratio: Tỉ lệ chia tập huấn luyện theo trục thời gian.
			max_pacf_lags: Số lag tối đa dùng khi ước lượng PACF.
			pacf_method: Phương pháp tính PACF của statsmodels.
			rolling_windows: Danh sách cửa sổ để tạo thống kê trượt.
			decomposition_period: Chu kỳ mùa vụ cho phân rã chuỗi.
			random_state: Hạt giống ngẫu nhiên cho mô hình.
			n_estimators: Số cây của RandomForestRegressor.
			max_depth: Độ sâu tối đa của cây (None là không giới hạn).
			min_samples_leaf: Số mẫu tối thiểu ở mỗi nút lá.
			n_jobs: Số luồng xử lý song song cho mô hình.
			model_kwargs: Tham số bổ sung truyền thẳng vào mô hình.

		Output:
			None.
		"""
		if forecast_horizon <= 0:
			raise ValueError("forecast_horizon must be >= 1")
		if not (0 < train_ratio < 1):
			raise ValueError("train_ratio must be in (0, 1)")
		if max_pacf_lags <= 0:
			raise ValueError("max_pacf_lags must be >= 1")
		if decomposition_period <= 1:
			raise ValueError("decomposition_period must be >= 2")

		windows = sorted({int(w) for w in rolling_windows if int(w) > 0})
		if not windows:
			raise ValueError("rolling_windows must contain at least one positive integer")

		self.target_column = target_column
		self.time_column = time_column
		self.entity_column = entity_column
		self.entity_value = entity_value
		self.aggregation = aggregation

		self.forecast_horizon = forecast_horizon
		self.train_ratio = train_ratio
		self.max_pacf_lags = max_pacf_lags
		self.pacf_method = pacf_method
		self.rolling_windows = windows
		self.decomposition_period = decomposition_period

		self.random_state = random_state
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.min_samples_leaf = min_samples_leaf
		self.n_jobs = n_jobs
		self.model_kwargs = model_kwargs or {}

		self._step_name = "Feature Matrix + RandomForest Forecast (t+1)"
		self._dataset_name = "Unknown"
		self._status = "Initialized"
		self._error_message = ""

		self._raw_series_df: pd.DataFrame | None = None
		self._supervised_df: pd.DataFrame | None = None
		self._feature_columns: list[str] = []
		self._target_name = f"target_t_plus_{self.forecast_horizon}"

		self.lag_order_: int | None = None
		self.pacf_table_: pd.DataFrame | None = None

		self.model_: RandomForestRegressor | None = None
		self.metrics_: dict[str, float] = {}
		self.test_result_: pd.DataFrame | None = None

	@property
	def feature_columns(self) -> list[str]:
		"""
		Lấy danh sách cột đặc trưng được dùng để huấn luyện mô hình.

		Input:
			Không có.

		Output:
			list[str]: Bản sao danh sách cột đặc trưng.
		"""
		return list(self._feature_columns)

	@property
	def feature_matrix(self) -> pd.DataFrame | None:
		"""
		Lấy ma trận giám sát đã xây dựng từ chuỗi thời gian.

		Input:
			Không có.

		Output:
			pd.DataFrame | None: Bản sao dữ liệu đặc trưng-target hoặc None nếu chưa có.
		"""
		return None if self._supervised_df is None else self._supervised_df.copy()

	def _prepare_univariate_series(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Chuẩn hóa dữ liệu đầu vào thành chuỗi đơn biến đã gộp theo thời gian.

		Input:
			df: Bảng dữ liệu gốc chứa cột thời gian và cột mục tiêu.

		Output:
			pd.DataFrame: Bảng gồm cột thời gian và cột target đã được làm sạch, lọc và tổng hợp.
		"""
		if self.time_column not in df.columns:
			raise ValueError(f"Missing time column: {self.time_column}")
		if self.target_column not in df.columns:
			raise ValueError(f"Missing target column: {self.target_column}")

		work_df = df.copy()
		work_df[self.time_column] = pd.to_datetime(work_df[self.time_column], errors="coerce")
		work_df = work_df.dropna(subset=[self.time_column, self.target_column])

		if (
			self.entity_column
			and self.entity_column in work_df.columns
			and self.entity_value
			and str(self.entity_value).lower() != "world"
		):
			work_df = work_df.loc[work_df[self.entity_column] == self.entity_value]
			if work_df.empty:
				raise ValueError(
					f"No rows found for {self.entity_column}={self.entity_value}."
				)

		grouped = (
			work_df.groupby(self.time_column, as_index=False)[self.target_column]
			.agg(self.aggregation)
			.sort_values(self.time_column)
			.reset_index(drop=True)
		)
		grouped.rename(columns={self.target_column: "target"}, inplace=True)

		if grouped.empty:
			raise ValueError("No valid rows available after preprocessing and aggregation.")

		return grouped

	def _infer_lag_order_from_pacf(self, series: pd.Series) -> int:
		"""
		Ước lượng bậc trễ tối ưu từ PACF dựa trên ngưỡng ý nghĩa xấp xỉ.

		Input:
			series: Chuỗi mục tiêu theo thời gian dùng để phân tích PACF.

		Output:
			int: Bậc trễ được chọn để tạo các lag features.
		"""
		clean_series = series.dropna().astype(float)
		n_obs = len(clean_series)
		if n_obs < 5:
			self.pacf_table_ = pd.DataFrame(
				{"lag": [0, 1], "pacf": [1.0, 0.0], "is_significant": [False, False]}
			)
			return 1

		max_lag_allowed = max(1, n_obs // 2 - 1)
		nlags = min(self.max_pacf_lags, max_lag_allowed)

		pacf_values = pacf(clean_series, nlags=nlags, method=self.pacf_method)
		conf = 1.96 / np.sqrt(n_obs)
		significance = [False] + [abs(v) > conf for v in pacf_values[1:]]

		significant_lags = [lag for lag in range(1, len(pacf_values)) if significance[lag]]
		lag_order = max(significant_lags) if significant_lags else 1

		self.pacf_table_ = pd.DataFrame(
			{
				"lag": np.arange(0, len(pacf_values)),
				"pacf": pacf_values,
				"is_significant": significance,
			}
		)
		return int(lag_order)

	def _decompose_series(self, series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
		"""
		Phân rã chuỗi thành các thành phần trend, seasonal và residual.

		Input:
			series: Chuỗi mục tiêu cần phân rã.

		Output:
			tuple[pd.Series, pd.Series, pd.Series]: Bộ ba (trend, seasonal, residual).
		"""
		target = series.astype(float)

		try:
			stl_result = STL(target, period=self.decomposition_period, robust=True).fit()
			trend = pd.Series(stl_result.trend, index=target.index)
			seasonal = pd.Series(stl_result.seasonal, index=target.index)
			resid = pd.Series(stl_result.resid, index=target.index)
			return trend, seasonal, resid
		except Exception:
			min_len = 2 * self.decomposition_period
			if len(target) >= min_len:
				dec = seasonal_decompose(
					target,
					model="additive",
					period=self.decomposition_period,
					extrapolate_trend="freq",
				)
				return dec.trend, dec.seasonal, dec.resid

			trend = target.rolling(window=self.decomposition_period, min_periods=1).mean()
			seasonal = target - trend
			resid = target - trend - seasonal
			return trend, seasonal, resid

	def _add_rolling_features(self, frame: pd.DataFrame) -> pd.DataFrame:
		"""
		Bổ sung các đặc trưng thống kê trượt dựa trên target trễ 1 bước.

		Input:
			frame: Bảng dữ liệu hiện tại chứa ít nhất cột target.

		Output:
			pd.DataFrame: Bảng dữ liệu mới có thêm các cột rolling mean/std/min/max.
		"""
		output = frame.copy()
		lagged_target = output["target"].shift(1)

		for window in self.rolling_windows:
			output[f"roll_mean_{window}"] = lagged_target.rolling(window=window, min_periods=1).mean()
			output[f"roll_std_{window}"] = lagged_target.rolling(window=window, min_periods=2).std()
			output[f"roll_min_{window}"] = lagged_target.rolling(window=window, min_periods=1).min()
			output[f"roll_max_{window}"] = lagged_target.rolling(window=window, min_periods=1).max()

		return output

	def _add_time_features(self, frame: pd.DataFrame) -> pd.DataFrame:
		"""
		Sinh các đặc trưng lịch thời gian và biến chu kỳ từ cột thời gian.

		Input:
			frame: Bảng dữ liệu chứa cột thời gian đã chuẩn hóa kiểu datetime.

		Output:
			pd.DataFrame: Bảng dữ liệu mới có thêm các cột đặc trưng theo thời gian.
		"""
		output = frame.copy()
		dt = output[self.time_column].dt

		output["year"] = dt.year
		output["month"] = dt.month
		output["day"] = dt.day
		output["dayofweek"] = dt.dayofweek
		output["dayofyear"] = dt.dayofyear
		output["weekofyear"] = dt.isocalendar().week.astype(int)
		output["quarter"] = dt.quarter
		output["is_month_start"] = dt.is_month_start.astype(int)
		output["is_month_end"] = dt.is_month_end.astype(int)
		output["is_weekend"] = dt.dayofweek.isin([5, 6]).astype(int)

		output["month_sin"] = np.sin(2.0 * np.pi * output["month"] / 12.0)
		output["month_cos"] = np.cos(2.0 * np.pi * output["month"] / 12.0)
		output["dow_sin"] = np.sin(2.0 * np.pi * output["dayofweek"] / 7.0)
		output["dow_cos"] = np.cos(2.0 * np.pi * output["dayofweek"] / 7.0)

		return output

	def _build_supervised_frame(self, grouped_df: pd.DataFrame, lag_order: int) -> pd.DataFrame:
		"""
		Tạo ma trận giám sát cuối cùng từ chuỗi đã gộp và bậc trễ được chọn.

		Input:
			grouped_df: Dữ liệu chuỗi đã gộp theo thời gian và có cột target.
			lag_order: Số bậc trễ dùng để sinh lag features.

		Output:
			pd.DataFrame: Bảng supervised đã có đặc trưng và nhãn dự báo tương lai.
		"""
		frame = grouped_df.copy()

		for lag in range(1, lag_order + 1):
			frame[f"lag_{lag}"] = frame["target"].shift(lag)

		frame = self._add_rolling_features(frame)
		frame = self._add_time_features(frame)

		trend, seasonal, resid = self._decompose_series(frame["target"])
		frame["decomp_trend_lag1"] = trend.shift(1)
		frame["decomp_seasonal_lag1"] = seasonal.shift(1)
		frame["decomp_resid_lag1"] = resid.shift(1)

		frame[self._target_name] = frame["target"].shift(-self.forecast_horizon)

		cleaned = frame.dropna().reset_index(drop=True)
		if cleaned.empty:
			raise ValueError(
				"Feature matrix is empty after dropna. Try fewer lags/windows or smaller decomposition_period."
			)

		return cleaned

	def _get_feature_columns(self, supervised_df: pd.DataFrame) -> list[str]:
		"""
		Lấy danh sách tên cột đặc trưng sau khi loại cột thời gian và nhãn đích.

		Input:
			supervised_df: Bảng supervised đã tạo đầy đủ đặc trưng.

		Output:
			list[str]: Danh sách tên cột dùng làm đầu vào mô hình.
		"""
		excluded = {self.time_column, "target", self._target_name}
		return [col for col in supervised_df.columns if col not in excluded]

	def _temporal_split(self, X: pd.DataFrame, y: pd.Series):
		"""
		Chia tập train/test theo thứ tự thời gian, không xáo trộn dữ liệu.

		Input:
			X: Ma trận đặc trưng đầu vào.
			y: Chuỗi nhãn mục tiêu tương ứng với X.

		Output:
			tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test.
		"""
		split_idx = int(len(X) * self.train_ratio)
		if split_idx <= 0 or split_idx >= len(X):
			raise ValueError("Cannot split train/test with current train_ratio and dataset length.")

		X_train = X.iloc[:split_idx].copy()
		X_test = X.iloc[split_idx:].copy()
		y_train = y.iloc[:split_idx].copy()
		y_test = y.iloc[split_idx:].copy()
		return X_train, X_test, y_train, y_test

	def fit(self, df: pd.DataFrame, y=None):
		"""
		Huấn luyện pipeline đặc trưng + Random Forest và tính metric trên tập test theo thời gian.

		Input:
			df: Dữ liệu chuỗi thời gian gốc dùng để huấn luyện.
			y: Tham số dự phòng theo chuẩn sklearn, không sử dụng trong pipeline này.

		Output:
			FeatureMatrixPredictionPreprocessing: Chính đối tượng hiện tại sau khi fit.
		"""
		self._raw_series_df = self._prepare_univariate_series(df)
		self.lag_order_ = self._infer_lag_order_from_pacf(self._raw_series_df["target"])

		self._supervised_df = self._build_supervised_frame(self._raw_series_df, lag_order=self.lag_order_)
		self._feature_columns = self._get_feature_columns(self._supervised_df)

		X = self._supervised_df[self._feature_columns]
		y_target = self._supervised_df[self._target_name]
		X_train, X_test, y_train, y_test = self._temporal_split(X, y_target)

		params = {
			"n_estimators": self.n_estimators,
			"max_depth": self.max_depth,
			"min_samples_leaf": self.min_samples_leaf,
			"random_state": self.random_state,
			"n_jobs": self.n_jobs,
		}
		params.update(self.model_kwargs)

		self.model_ = RandomForestRegressor(**params)
		self.model_.fit(X_train, y_train)

		y_pred = self.model_.predict(X_test)
		mae = mean_absolute_error(y_test, y_pred)
		rmse = np.sqrt(mean_squared_error(y_test, y_pred))

		self.metrics_ = {
			"mae": float(mae),
			"rmse": float(rmse),
			"train_size": float(len(X_train)),
			"test_size": float(len(X_test)),
			"lag_order": float(self.lag_order_),
			"feature_count": float(len(self._feature_columns)),
		}

		self.test_result_ = pd.DataFrame(
			{
				self.time_column: self._supervised_df.loc[X_test.index, self.time_column].to_numpy(),
				"y_true": y_test.to_numpy(),
				"y_pred": y_pred,
			}
		).reset_index(drop=True)

		return self

	def transform(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Biến đổi dữ liệu mới thành ma trận supervised theo cấu hình hiện tại.

		Input:
			df: Dữ liệu chuỗi thời gian cần chuyển thành đặc trưng.

		Output:
			pd.DataFrame: Bảng supervised đã tạo từ dữ liệu đầu vào.
		"""
		grouped = self._prepare_univariate_series(df)
		lag_order = self.lag_order_ if self.lag_order_ is not None else self._infer_lag_order_from_pacf(grouped["target"])
		return self._build_supervised_frame(grouped, lag_order=lag_order)

	def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
		"""
		Thực hiện fit rồi trả về trực tiếp ma trận supervised đã huấn luyện.

		Input:
			df: Dữ liệu chuỗi thời gian đầu vào.
			y: Tham số dự phòng theo chuẩn sklearn, không dùng trong xử lý.

		Output:
			pd.DataFrame: Bản sao ma trận supervised sau khi fit.
		"""
		self.fit(df, y)
		return self._supervised_df.copy()

	def run(self, obj):
		"""
		Điểm vào thực thi theo pattern visitor của hệ thống pipeline.

		Input:
			obj: Đối tượng dataset cần xử lý.

		Output:
			None.
		"""
		self.visitTableDataset(obj)

	def visitImageDataset(self, obj):
		"""
		Thông báo không hỗ trợ dataset ảnh trong tiền xử lý này.

		Input:
			obj: Đối tượng dataset ảnh (không được hỗ trợ).

		Output:
			None.
		"""
		print(f"[WARNING] {self.__class__.__name__} does not support ImageDataset.")

	def visitTableDataset(self, obj):
		"""
		Áp dụng pipeline lên dataset bảng và cập nhật lại các thuộc tính đầu ra trên dataset.

		Input:
			obj: Đối tượng dataset có thuộc tính data để huấn luyện và biến đổi.

		Output:
			None.
		"""
		self._dataset_name = getattr(obj, "_folder_path", "TimeSeriesDataset")

		try:
			if not hasattr(obj, "data") or obj.data is None:
				raise ValueError("Dataset does not contain data. Load dataset first.")

			self.fit(obj.data)

			obj._data = self._supervised_df.copy()
			obj._features = self._supervised_df[self._feature_columns].copy()
			obj._target_column = self._target_name
			obj._target = self._supervised_df[self._target_name].copy()

			self._status = "Success"
			self._error_message = ""
		except Exception as exc:
			self._status = "Failed"
			self._error_message = str(exc)
			raise
		finally:
			self.log()

	def get_metrics(self) -> dict[str, float]:
		"""
		Lấy bộ chỉ số đánh giá hiện tại của mô hình dự báo.

		Input:
			Không có.

		Output:
			dict[str, float]: Bản sao các metric như MAE, RMSE, lag_order, feature_count.
		"""
		return dict(self.metrics_)

	def log(self):
		"""
		In nhật ký tóm tắt trạng thái xử lý và kết quả đánh giá.

		Input:
			Không có.

		Output:
			None.
		"""
		print("\n" + "=" * 60)
		print(f"Step       : {self._step_name}")
		print(f"Dataset    : {self._dataset_name}")
		print(f"Status     : {self._status}")

		if self._status == "Success":
			print(f"Lag order  : {self.lag_order_}")
			print(f"Features   : {len(self._feature_columns)}")
			if self.metrics_:
				print(f"MAE (test) : {self.metrics_['mae']:.6f}")
				print(f"RMSE(test) : {self.metrics_['rmse']:.6f}")
		else:
			print(f"Error      : {self._error_message}")

		print("=" * 60)
