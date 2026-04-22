"""Advanced categorical encoding for tabular datasets."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from core.service_base import Preprocessing


class _BaseCategoricalEncoding(Preprocessing):
    """Shared workflow for categorical encoders used on TableDataset."""

    def __init__(self, step_name: str, columns: list[str] | None = None, drop_original: bool = True):
        """
        Khởi tạo encoder cơ sở.

        Input:
            step_name: Tên bước xử lý.
            columns: Danh sách cột cần mã hóa (None = tự phát hiện cột categorical).
            drop_original: True nếu xóa cột gốc sau mã hóa.

        Output:
            None.
        """
        self._step_name = step_name
        self._columns = columns
        self._drop_original = drop_original

        self._status = "Initialized"
        self._error_message = ""
        self._dataset_path = "Unknown"

        self._is_fitted = False
        self._resolved_columns: list[str] = []
        self._feature_names_out: list[str] = []
        self._transformed: pd.DataFrame | np.ndarray | None = None
        self._vif_report = pd.DataFrame(columns=["feature", "vif"])

    @property
    def resolved_columns(self) -> list[str]:
        return self._resolved_columns

    @property
    def feature_names_out(self) -> list[str]:
        return self._feature_names_out

    @property
    def transformed_features(self) -> pd.DataFrame | np.ndarray | None:
        return self._transformed

    @property
    def vif_report(self) -> pd.DataFrame:
        return self._vif_report.copy()

    def _to_dataframe(self, X: Any) -> tuple[pd.DataFrame, bool]:
        """
        Chuyển đổi đầu vào thành DataFrame.

        Input:
            X: Dữ liệu đầu vào (DataFrame hoặc numpy array 2D).

        Output:
            Tuple (DataFrame, bool): DataFrame đã chuyển đổi và cờ is_dataframe.
        """
        if isinstance(X, pd.DataFrame):
            return X.copy(), True

        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError("Input must be a 2D numpy array or pandas DataFrame.")

        columns = [f"feature_{i}" for i in range(X_arr.shape[1])]
        return pd.DataFrame(X_arr, columns=columns), False

    def _coerce_target(self, y: Any, index: pd.Index, required: bool = False) -> pd.Series | None:
        """
        Chuyển đổi target thành pd.Series với index khớp.

        Input:
            y: Biến mục tiêu (Series, array, hoặc None).
            index: Index từ DataFrame X.
            required: True nếu bắt buộc phải có target.

        Output:
            pd.Series | None: Target đã chuẩn hóa hoặc None.
        """
        if y is None:
            if required:
                raise ValueError("This encoder requires target values y.")
            return None

        if isinstance(y, pd.Series):
            y_series = y.reindex(index)
        else:
            y_arr = np.asarray(y)
            if y_arr.ndim != 1:
                raise ValueError("Target y must be a 1D vector.")
            if y_arr.shape[0] != len(index):
                raise ValueError("Length of target y does not match number of rows in X.")
            y_series = pd.Series(y_arr, index=index, name="target")

        if y_series.shape[0] != len(index):
            raise ValueError("Length of target y does not match number of rows in X.")

        return y_series

    def _resolve_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Xác định danh sách cột categorical cần mã hóa.

        Input:
            df: DataFrame đầu vào.

        Output:
            list[str]: Danh sách tên cột đã xác định.
        """
        if self._columns is None:
            return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        missing = [col for col in self._columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in input data: {missing}")
        return list(self._columns)

    @staticmethod
    def _normalize_category(series: pd.Series) -> pd.Series:
        """
        Chuẩn hóa cột categorical: ép string và điền giá trị thiếu.

        Input:
            series: Series dữ liệu categorical.

        Output:
            pd.Series: Series đã chuẩn hóa.
        """
        return series.astype("string").fillna("__MISSING__")

    @staticmethod
    def _compute_vif(df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
        """
        Tính Variance Inflation Factor (VIF) cho các cột số.

        Input:
            df: DataFrame chứa dữ liệu đã mã hóa.
            eps: Hằng số tránh chia cho 0.

        Output:
            pd.DataFrame: Bảng VIF sắp xếp giảm dần theo giá trị VIF.
        """
        numeric = df.select_dtypes(include=[np.number]).copy()
        if numeric.shape[1] < 2:
            return pd.DataFrame(columns=["feature", "vif"])

        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        numeric = numeric.fillna(numeric.median(numeric_only=True))
        numeric = numeric.fillna(0.0)

        X = numeric.to_numpy(dtype=float)
        names = numeric.columns.tolist()
        vif_rows: list[dict[str, float | str]] = []

        for i, feature_name in enumerate(names):
            y_col = X[:, i]
            x_others = np.delete(X, i, axis=1)

            if x_others.shape[1] == 0:
                vif_value = 1.0
            elif np.var(y_col) <= eps:
                vif_value = np.inf
            else:
                design = np.column_stack([np.ones(len(y_col)), x_others])
                coef, _, _, _ = np.linalg.lstsq(design, y_col, rcond=None)
                y_pred = design @ coef

                ss_res = float(np.sum((y_col - y_pred) ** 2))
                ss_tot = float(np.sum((y_col - np.mean(y_col)) ** 2))

                if ss_tot <= eps:
                    vif_value = np.inf
                else:
                    r2 = 1.0 - (ss_res / ss_tot)
                    if r2 >= 1.0 - eps:
                        vif_value = np.inf
                    else:
                        vif_value = 1.0 / max(1.0 - r2, eps)

            vif_rows.append({"feature": feature_name, "vif": float(vif_value) if np.isfinite(vif_value) else np.inf})

        return pd.DataFrame(vif_rows).sort_values("vif", ascending=False).reset_index(drop=True)

    def _fit_encoder(self, X: pd.DataFrame, y: pd.Series | None):
        """
        Logic fit cụ thể (lớp con phải implement).

        Input:
            X: DataFrame features.
            y: Series target (có thể None).

        Output:
            None.
        """
        raise NotImplementedError

    def _transform_encoder(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Logic transform cụ thể (lớp con phải implement).

        Input:
            X: DataFrame features.

        Output:
            pd.DataFrame: Dữ liệu đã mã hóa.
        """
        raise NotImplementedError

    def fit(self, X, y=None):
        """
        Tính toán mapping từ dữ liệu huấn luyện.

        Input:
            X: Dữ liệu features (DataFrame hoặc numpy array).
            y: Biến mục tiêu (tùy chọn).

        Output:
            self: Trả về chính đối tượng encoder.
        """
        X_df, _ = self._to_dataframe(X)
        y_series = self._coerce_target(y, X_df.index, required=False)

        self._resolved_columns = self._resolve_columns(X_df)
        self._fit_encoder(X_df, y_series)

        self._is_fitted = True
        return self

    def transform(self, X):
        """
        Áp dụng mã hóa lên dữ liệu dựa trên mapping đã fit.

        Input:
            X: Dữ liệu features cần mã hóa.

        Output:
            pd.DataFrame hoặc np.ndarray: Dữ liệu đã mã hóa.
        """
        if not self._is_fitted:
            raise ValueError("Encoder is not fitted. Call fit before transform.")

        X_df, is_dataframe = self._to_dataframe(X)
        transformed_df = self._transform_encoder(X_df)

        self._feature_names_out = transformed_df.columns.astype(str).tolist()
        self._vif_report = self._compute_vif(transformed_df)
        self._transformed = transformed_df

        return transformed_df if is_dataframe else transformed_df.to_numpy()

    def fit_transform(self, X, y=None):
        """
        Kết hợp fit và transform trong một bước.

        Input:
            X: Dữ liệu features.
            y: Biến mục tiêu (tùy chọn).

        Output:
            pd.DataFrame hoặc np.ndarray: Dữ liệu đã mã hóa.
        """
        self.fit(X, y)
        return self.transform(X)

    def visitImageDataset(self, obj):
        """
        Không hỗ trợ dữ liệu hình ảnh.

        Input:
            obj: Đối tượng ImageDataset.

        Output:
            None (in cảnh báo).
        """
        print(f"[WARNING] {self.__class__.__name__} does not support ImageDataset.")
        return

    def visitTableDataset(self, obj):
        """
        Triển khai mã hóa lên đối tượng TableDataset.

        Input:
            obj: Đối tượng TableDataset chứa dữ liệu cần mã hóa.

        Output:
            None (cập nhật trực tiếp dữ liệu trong obj).
        """
        self._dataset_path = getattr(obj, "_folder_path", "Unknown")

        try:
            X = getattr(obj, "features", None)
            y = getattr(obj, "target", None)
            if X is None:
                raise ValueError(
                    "Dataset does not have features. Call set_target(target_column) before preprocessing."
                )

            X_encoded = self.fit_transform(X, y)
            if not isinstance(X_encoded, pd.DataFrame):
                X_encoded = pd.DataFrame(
                    X_encoded,
                    columns=self._feature_names_out,
                    index=getattr(X, "index", None),
                )

            obj._features = X_encoded

            target_col = getattr(obj, "_target_column", None)
            if target_col is not None and y is not None:
                y_series = y if isinstance(y, pd.Series) else pd.Series(y, name=target_col, index=X_encoded.index)
                obj.data = pd.concat([X_encoded, y_series.rename(target_col)], axis=1)
                obj.set_target(target_col)
            else:
                obj.data = X_encoded

            self._status = "Success"
        except Exception as exc:
            self._status = "Failed"
            self._error_message = str(exc)

    def run(self, obj):
        """
        Điểm vào thực thi - gọi visitTableDataset.

        Input:
            obj: Đối tượng TableDataset.

        Output:
            None.
        """
        self.visitTableDataset(obj)

    def log(self):
        """
        In thông tin trạng thái và kết quả mã hóa.

        Input:
            Không có.

        Output:
            None (in ra màn hình).
        """
        print("\n" + "=" * 60)
        print(f"Step      : {self._step_name}")
        print(f"Dataset   : {self._dataset_path}")
        print(f"Status    : {self._status}")
        if self._status == "Success":
            print(f"Columns   : {self._resolved_columns}")
            if self._vif_report.empty:
                print("VIF       : Not available (need at least 2 numeric columns).")
            else:
                print(f"VIF max   : {self._vif_report['vif'].max():.6f}")
        else:
            print(f"Error     : {self._error_message}")
        print("=" * 60)


class TargetEncodingCV(_BaseCategoricalEncoding):
    """Mean target encoding using out-of-fold strategy to reduce target leakage."""

    def __init__(
        self,
        columns: list[str] | None = None,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        smoothing: float = 10.0,
        suffix: str = "_target",
        drop_original: bool = True,
    ):
        """
        Khởi tạo Target Encoding với chiến lược out-of-fold (CV).

        Input:
            columns: Danh sách cột cần mã hóa (None = tự phát hiện).
            n_splits: Số folds cho KFold (mặc định 5).
            shuffle: True nếu xáo trộn dữ liệu.
            random_state: Seed cho random.
            smoothing: Hệ số smoothing giảm target leakage.
            suffix: Hậu tố cột mã hóa.
            drop_original: True nếu xóa cột gốc.

        Output:
            None.
        """
        super().__init__(
            step_name="Target Encoding (CV Mean)",
            columns=columns,
            drop_original=drop_original,
        )

        if n_splits < 2:
            raise ValueError("n_splits must be >= 2 for target encoding with CV.")
        if smoothing < 0:
            raise ValueError("smoothing must be >= 0.")

        self._n_splits = n_splits
        self._shuffle = shuffle
        self._random_state = random_state
        self._smoothing = smoothing
        self._suffix = suffix

        self._global_mean = 0.0
        self._mapping: dict[str, pd.Series] = {}

    def _require_numeric_target(self, y: pd.Series | None, index: pd.Index) -> pd.Series:
        """
        Đảm bảo target là số và không có giá trị thiếu.

        Input:
            y: Biến mục tiêu.
            index: Index DataFrame.

        Output:
            pd.Series: Target dạng số.
        """
        y_series = self._coerce_target(y, index, required=True)
        y_num = pd.to_numeric(y_series, errors="coerce")
        if y_num.isna().any():
            raise ValueError("Target Encoding requires numeric target y without missing values.")
        return y_num

    def _build_mapping(self, category_series: pd.Series, y: pd.Series) -> pd.Series:
        """
        Xây dựng bảng mapping từ category sang giá trị trung bình target (có smoothing).

        Input:
            category_series: Series chứa giá trị categorical.
            y: Series target số.

        Output:
            pd.Series: Mapping từ category sang giá trị đã mã hóa.
        """
        safe_cat = self._normalize_category(category_series)
        stats = pd.DataFrame({"target": y, "cat": safe_cat}).groupby("cat")["target"].agg(["mean", "count"])

        if self._smoothing == 0:
            return stats["mean"]

        smooth = (stats["mean"] * stats["count"] + self._global_mean * self._smoothing) / (
            stats["count"] + self._smoothing
        )
        return smooth

    def _fit_encoder(self, X: pd.DataFrame, y: pd.Series | None):
        """
        Tính global mean và xây dựng mapping target encoding.

        Input:
            X: DataFrame features.
            y: Series target.

        Output:
            None.
        """
        y_num = self._require_numeric_target(y, X.index)
        self._global_mean = float(y_num.mean())
        self._mapping = {}

        for col in self._resolved_columns:
            self._mapping[col] = self._build_mapping(X[col], y_num)

    def _transform_encoder(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Áp dụng target encoding mapping.

        Input:
            X: DataFrame features.

        Output:
            pd.DataFrame: Dữ liệu đã target encoding.
        """
        transformed = X.copy()

        for col in self._resolved_columns:
            encoded_col = f"{col}{self._suffix}"
            safe_cat = self._normalize_category(transformed[col])
            transformed[encoded_col] = safe_cat.map(self._mapping[col]).fillna(self._global_mean).astype(float)

            if self._drop_original:
                transformed = transformed.drop(columns=[col])

        return transformed

    def fit_transform(self, X, y=None):
        X_df, is_dataframe = self._to_dataframe(X)
        y_num = self._require_numeric_target(y, X_df.index)
        self._resolved_columns = self._resolve_columns(X_df)
        self._global_mean = float(y_num.mean())

        n_splits = min(self._n_splits, len(X_df))
        if n_splits < 2:
            raise ValueError("Need at least 2 rows for cross-validated target encoding.")

        kf = KFold(n_splits=n_splits, shuffle=self._shuffle, random_state=self._random_state)
        transformed = X_df.copy()

        for col in self._resolved_columns:
            encoded_col = f"{col}{self._suffix}"
            oof_encoded = pd.Series(np.nan, index=X_df.index, dtype=float)

            for train_idx, valid_idx in kf.split(X_df):
                x_train = X_df.iloc[train_idx]
                y_train = y_num.iloc[train_idx]
                fold_global = float(y_train.mean())

                safe_train = self._normalize_category(x_train[col])
                stats = pd.DataFrame({"target": y_train, "cat": safe_train}).groupby("cat")["target"].agg(["mean", "count"])

                if self._smoothing == 0:
                    fold_mapping = stats["mean"]
                else:
                    fold_mapping = (
                        (stats["mean"] * stats["count"] + fold_global * self._smoothing)
                        / (stats["count"] + self._smoothing)
                    )

                safe_valid = self._normalize_category(X_df.iloc[valid_idx][col])
                oof_encoded.iloc[valid_idx] = safe_valid.map(fold_mapping).fillna(fold_global)

            transformed[encoded_col] = oof_encoded.fillna(self._global_mean).astype(float)
            if self._drop_original:
                transformed = transformed.drop(columns=[col])

        self._fit_encoder(X_df, y_num)
        self._is_fitted = True
        self._feature_names_out = transformed.columns.astype(str).tolist()
        self._vif_report = self._compute_vif(transformed)
        self._transformed = transformed

        return transformed if is_dataframe else transformed.to_numpy()


class BinaryEncoding(_BaseCategoricalEncoding):
    """Binary encoding for high-cardinality categorical columns."""

    def __init__(
        self,
        columns: list[str] | None = None,
        min_cardinality: int = 20,
        drop_original: bool = True,
    ):
        """
        Khởi tạo Binary Encoding cho cột categorical có cardinality cao.

        Input:
            columns: Danh sách cột cần mã hóa.
            min_cardinality: Ngưỡng cardinality tối thiểu để áp dụng.
            drop_original: True nếu xóa cột gốc.

        Output:
            None.
        """
        super().__init__(
            step_name="Binary Encoding",
            columns=columns,
            drop_original=drop_original,
        )
        if min_cardinality < 1:
            raise ValueError("min_cardinality must be >= 1.")

        self._min_cardinality = min_cardinality
        self._active_columns: list[str] = []
        self._mapping: dict[str, dict[str, int]] = {}
        self._n_bits: dict[str, int] = {}

    def _fit_encoder(self, X: pd.DataFrame, y: pd.Series | None):
        self._active_columns = []
        self._mapping = {}
        self._n_bits = {}

        for col in self._resolved_columns:
            safe_cat = self._normalize_category(X[col])
            cardinality = int(safe_cat.nunique(dropna=False))

            if cardinality <= self._min_cardinality:
                continue

            categories = sorted(safe_cat.unique().tolist())
            mapping = {cat: idx + 1 for idx, cat in enumerate(categories)}
            n_bits = max(1, math.ceil(math.log2(len(mapping) + 1)))

            self._active_columns.append(col)
            self._mapping[col] = mapping
            self._n_bits[col] = n_bits

    def _transform_encoder(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = X.copy()

        for col in self._active_columns:
            safe_cat = self._normalize_category(transformed[col])
            code = safe_cat.map(self._mapping[col]).fillna(0).astype(int)
            code_values = code.to_numpy(dtype=int)

            bit_count = self._n_bits[col]
            for bit in range(bit_count):
                shift = bit_count - bit - 1
                bit_col = f"{col}_bin_{bit + 1}"
                transformed[bit_col] = ((code_values >> shift) & 1).astype(int)

            if self._drop_original:
                transformed = transformed.drop(columns=[col])

        return transformed


class FrequencyEncoding(_BaseCategoricalEncoding):
    """Frequency encoding for categorical columns."""

    def __init__(
        self,
        columns: list[str] | None = None,
        suffix: str = "_freq",
        drop_original: bool = True,
    ):
        """
        Khởi tạo Frequency Encoding.

        Input:
            columns: Danh sách cột cần mã hóa.
            suffix: Hậu tố cột mã hóa.
            drop_original: True nếu xóa cột gốc.

        Output:
            None.
        """
        super().__init__(
            step_name="Frequency Encoding",
            columns=columns,
            drop_original=drop_original,
        )
        self._suffix = suffix
        self._mapping: dict[str, pd.Series] = {}

    def _fit_encoder(self, X: pd.DataFrame, y: pd.Series | None):
        self._mapping = {}
        for col in self._resolved_columns:
            safe_cat = self._normalize_category(X[col])
            self._mapping[col] = safe_cat.value_counts(normalize=True, dropna=False)

    def _transform_encoder(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = X.copy()

        for col in self._resolved_columns:
            encoded_col = f"{col}{self._suffix}"
            safe_cat = self._normalize_category(transformed[col])
            transformed[encoded_col] = safe_cat.map(self._mapping[col]).fillna(0.0).astype(float)

            if self._drop_original:
                transformed = transformed.drop(columns=[col])

        return transformed


class OneHotEncoding(_BaseCategoricalEncoding):
    """One-hot encoding for categorical columns."""

    def __init__(
        self,
        columns: list[str] | None = None,
        drop_first: bool = False,
        drop_original: bool = True,
        prefix_sep: str = "_",
    ):
        """
        Khởi tạo One-Hot Encoding.

        Input:
            columns: Danh sách cột cần mã hóa.
            drop_first: True nếu bỏ cột đầu tiên (tránh multicollinearity).
            drop_original: True nếu xóa cột gốc.
            prefix_sep: Ký tự ngăn cách prefix.

        Output:
            None.
        """
        super().__init__(
            step_name="One-Hot Encoding",
            columns=columns,
            drop_original=drop_original,
        )
        self._drop_first = drop_first
        self._prefix_sep = prefix_sep
        self._categories: dict[str, list[str]] = {}

    def _fit_encoder(self, X: pd.DataFrame, y: pd.Series | None):
        self._categories = {}
        for col in self._resolved_columns:
            safe_cat = self._normalize_category(X[col])
            self._categories[col] = sorted(safe_cat.unique().tolist())

    def _transform_encoder(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = X.copy()

        for col in self._resolved_columns:
            safe_cat = self._normalize_category(transformed[col])
            cat_series = pd.Categorical(safe_cat, categories=self._categories[col])

            dummies = pd.get_dummies(
                cat_series,
                prefix=col,
                prefix_sep=self._prefix_sep,
                dtype=int,
            )

            if self._drop_first and dummies.shape[1] > 0:
                dummies = dummies.iloc[:, 1:]

            insert_at = transformed.columns.get_loc(col) + 1
            transformed = pd.concat(
                [transformed.iloc[:, :insert_at], dummies, transformed.iloc[:, insert_at:]],
                axis=1,
            )

            if self._drop_original:
                transformed = transformed.drop(columns=[col])

        return transformed


class OrdinalEncoding(_BaseCategoricalEncoding):
    """Ordinal encoding for categorical columns."""

    def __init__(
        self,
        columns: list[str] | None = None,
        drop_original: bool = True,
        unknown_value: int = -1,
        suffix: str = "_ord",
    ):
        """
        Khởi tạo Ordinal Encoding.

        Input:
            columns: Danh sách cột cần mã hóa.
            drop_original: True nếu xóa cột gốc.
            unknown_value: Giá trị gán cho category không biết (mặc định -1).
            suffix: Hậu tố cột mã hóa.

        Output:
            None.
        """
        super().__init__(
            step_name="Ordinal Encoding",
            columns=columns,
            drop_original=drop_original,
        )
        self._unknown_value = unknown_value
        self._suffix = suffix
        self._mapping: dict[str, dict[str, int]] = {}

    def _fit_encoder(self, X: pd.DataFrame, y: pd.Series | None):
        self._mapping = {}
        for col in self._resolved_columns:
            safe_cat = self._normalize_category(X[col])
            categories = sorted(safe_cat.unique().tolist())
            self._mapping[col] = {cat: idx for idx, cat in enumerate(categories)}

    def _transform_encoder(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = X.copy()

        for col in self._resolved_columns:
            encoded_col = f"{col}{self._suffix}"
            safe_cat = self._normalize_category(transformed[col])
            transformed[encoded_col] = safe_cat.map(self._mapping[col]).fillna(self._unknown_value).astype(int)

            if self._drop_original:
                transformed = transformed.drop(columns=[col])

        return transformed


class EncodingComparison:
    """Compare multiple encoders and summarize VIF impact."""

    def __init__(self, vif_threshold: float = 10.0):
        """
        Khởi tạo bộ so sánh encoder.

        Input:
            vif_threshold: Ngưỡng VIF để xác định cột có đa cộng tuyến cao.

        Output:
            None.
        """
        if vif_threshold <= 0:
            raise ValueError("vif_threshold must be > 0.")
        self._vif_threshold = vif_threshold
        self._summary = pd.DataFrame()

    @property
    def summary(self) -> pd.DataFrame:
        return self._summary.copy()

    def compare(self, X, y, encoders: dict[str, _BaseCategoricalEncoding]) -> pd.DataFrame:
        """
        Chạy nhiều encoder và so sánh kết quả VIF.

        Input:
            X: Dữ liệu features.
            y: Biến mục tiêu.
            encoders: Dict mapping tên phương pháp -> đối tượng encoder.

        Output:
            pd.DataFrame: Bảng tổng hợp so sánh các encoder.
        """
        rows: list[dict[str, float | int | str]] = []

        for method_name, encoder in encoders.items():
            transformed = encoder.fit_transform(X, y)
            transformed_df = (
                transformed
                if isinstance(transformed, pd.DataFrame)
                else pd.DataFrame(transformed, columns=encoder.feature_names_out)
            )

            vif_df = encoder.vif_report
            if vif_df.empty:
                max_vif = np.nan
                mean_vif = np.nan
                high_vif_count = 0
            else:
                max_vif = float(vif_df["vif"].replace(np.inf, np.nan).max())
                mean_vif = float(vif_df["vif"].replace(np.inf, np.nan).mean())
                high_vif_count = int((vif_df["vif"] >= self._vif_threshold).sum())

            rows.append(
                {
                    "method": method_name,
                    "n_features_after_encoding": int(transformed_df.shape[1]),
                    "max_vif": max_vif,
                    "mean_vif": mean_vif,
                    "n_high_vif": high_vif_count,
                }
            )

        self._summary = pd.DataFrame(rows).sort_values("n_high_vif", ascending=False).reset_index(drop=True)
        return self._summary
