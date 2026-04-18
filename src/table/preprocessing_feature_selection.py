"""Feature selection and dimensionality reduction for tabular datasets."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from core.service_base import Preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score

from visualization.relationship import plot_dim_reduction_2d


class _BaseFeaturePreprocessing(Preprocessing):
    def __init__(self, step_name: str, k: int | None = None):
        self._step_name = step_name
        self._k = k
        self._status = "Initialized"
        self._error_message = ""
        self._dataset_path = "Unknown"

        self._selected_indices: np.ndarray | None = None
        self._selected_feature_names: list[str] = []
        self._X_selected: np.ndarray | None = None

    @property
    def selected_indices(self) -> np.ndarray | None:
        return self._selected_indices

    @property
    def selected_feature_names(self) -> list[str]:
        return self._selected_feature_names

    @property
    def transformed_features(self) -> np.ndarray | None:
        return self._X_selected

    def _ensure_xy(self, obj: Any) -> tuple[Any, Any]:
        X = getattr(obj, "features", None)
        y = getattr(obj, "target", None)

        if X is None or y is None:
            raise ValueError(
                "Dataset chưa có features/target. Hãy gọi set_target(target_column) trước khi preprocessing."
            )
        return X, y

    def _cap_k(self, n_features: int) -> int:
        if self._k is None:
            raise ValueError("Thiếu tham số k.")
        if self._k <= 0:
            raise ValueError("k phải lớn hơn 0.")
        return min(self._k, n_features)

    def _infer_feature_names(self, X: Any) -> list[str]:
        if isinstance(X, pd.DataFrame):
            return X.columns.astype(str).tolist()
        return [f"feature_{i}" for i in range(np.asarray(X).shape[1])]

    def _to_selected_dataframe(self, X: Any) -> pd.DataFrame:
        if self._X_selected is None:
            raise ValueError("Chưa có dữ liệu transform.")
        return pd.DataFrame(self._X_selected, columns=self._selected_feature_names, index=getattr(X, "index", None))

    def visitImageDataset(self, obj):
        print(f"[WARNING] {self.__class__.__name__} không hỗ trợ ImageDataset.")
        return

    def visitTableDataset(self, obj):
        self._dataset_path = getattr(obj, "_folder_path", "Unknown")
        try:
            X, y = self._ensure_xy(obj)
            self.fit(X, y)
            X_new = self.transform(X)

            selected_df = pd.DataFrame(
                X_new,
                columns=self._selected_feature_names,
                index=getattr(X, "index", None),
            )
            obj._features = selected_df

            target_col = getattr(obj, "_target_column", None)
            if target_col:
                y_series = y if isinstance(y, pd.Series) else pd.Series(y, name=target_col)
                obj.data = pd.concat([selected_df, y_series.rename(target_col)], axis=1)
                obj.set_target(target_col)

            self._status = "Success"
        except Exception as e:
            self._status = "Failed"
            self._error_message = str(e)

    def run(self, obj):
        self.visitTableDataset(obj)

    def log(self):
        print("\n" + "=" * 55)
        print(f"Step      : {self._step_name}")
        print(f"Dataset   : {self._dataset_path}")
        print(f"Status    : {self._status}")
        if self._status == "Success":
            print(f"Selected  : {len(self._selected_feature_names)} feature(s)")
            print(f"Features  : {self._selected_feature_names}")
        else:
            print(f"Error     : {self._error_message}")
        print("=" * 55)


class StatisticalFiltering(_BaseFeaturePreprocessing):
    def __init__(self, k: int, score_func, method_name: str):
        super().__init__(step_name=f"Statistical Filtering ({method_name})", k=k)
        self._score_func = score_func
        self._method_name = method_name
        self._selector: SelectKBest | None = None
        self._scores: np.ndarray | None = None

    @property
    def scores(self) -> np.ndarray | None:
        return self._scores

    def _prepare_X(self, X: Any) -> np.ndarray:
        return np.asarray(X)

    def fit(self, X, y):
        X_arr = self._prepare_X(X)
        k = self._cap_k(X_arr.shape[1])
        self._selector = SelectKBest(score_func=self._score_func, k=k)
        self._selector.fit(X_arr, y)
        self._selected_indices = self._selector.get_support(indices=True)
        feature_names = self._infer_feature_names(X)
        self._selected_feature_names = [feature_names[i] for i in self._selected_indices]
        self._scores = self._selector.scores_

    def transform(self, X):
        if self._selector is None:
            raise ValueError("Model chưa fit. Hãy gọi fit trước.")
        X_arr = self._prepare_X(X)
        self._X_selected = self._selector.transform(X_arr)
        return self._X_selected

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class ANOVAFiltering(StatisticalFiltering):
    def __init__(self, k: int):
        super().__init__(k=k, score_func=f_classif, method_name="ANOVA")


class ChiSquareFiltering(StatisticalFiltering):
    def __init__(self, k: int):
        super().__init__(k=k, score_func=chi2, method_name="Chi-Square")
        self._offset: np.ndarray | None = None

    def _prepare_X(self, X: Any) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if self._offset is None:
            self._offset = np.maximum(0.0, -np.nanmin(X_arr, axis=0))
        return X_arr + self._offset


class MutualInformationFiltering(StatisticalFiltering):
    def __init__(self, k: int):
        score_func = partial(mutual_info_classif, random_state=42)
        super().__init__(k=k, score_func=score_func, method_name="Mutual Information")


class ModelBasedFiltering(_BaseFeaturePreprocessing):
    def __init__(self, k: int, model, method_name: str):
        super().__init__(step_name=f"Model-based Filtering ({method_name})", k=k)
        self._model = model
        self._method_name = method_name
        self._importances: np.ndarray | None = None

    @property
    def importances(self) -> np.ndarray | None:
        return self._importances

    def fit(self, X, y):
        X_arr = np.asarray(X)
        self._model.fit(X_arr, y)
        self._importances = self._model.feature_importances_

        k = self._cap_k(X_arr.shape[1])
        selected = np.argsort(self._importances)[::-1][:k]
        self._selected_indices = np.sort(selected)
        feature_names = self._infer_feature_names(X)
        self._selected_feature_names = [feature_names[i] for i in self._selected_indices]

    def transform(self, X):
        if self._selected_indices is None:
            raise ValueError("Model chưa fit. Hãy gọi fit trước.")
        X_arr = np.asarray(X)
        self._X_selected = X_arr[:, self._selected_indices]
        return self._X_selected

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class RandomForestFiltering(ModelBasedFiltering):
    def __init__(self, k: int, n_estimators: int = 100, random_state: int = 42):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        super().__init__(k=k, model=model, method_name="RandomForest")


class GradientBoostingFiltering(ModelBasedFiltering):
    def __init__(self, k: int, n_estimators: int = 100, random_state: int = 42):
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
        )
        super().__init__(k=k, model=model, method_name="GradientBoosting")


class DimensionReduction(Preprocessing):
    def __init__(self, n_components: int | float):
        self._step_name = f"Dimension Reduction (PCA: n_components={n_components})"
        self._n_components = n_components
        self._status = "Initialized"
        self._error_message = ""
        self._dataset_path = "Unknown"

        self._pca: PCA | None = None
        self._X_input: np.ndarray | None = None
        self._X_reduced: np.ndarray | None = None
        self._component_names: list[str] = []

    @property
    def pca(self) -> PCA | None:
        return self._pca

    @property
    def transformed_features(self) -> np.ndarray | None:
        return self._X_reduced

    def fit(self, X, y=None):
        self._X_input = np.asarray(X)
        self._pca = PCA(n_components=self._n_components, random_state=42)
        self._pca.fit(self._X_input)

    def transform(self, X):
        if self._pca is None:
            raise ValueError("PCA chưa fit. Hãy gọi fit trước.")
        X_arr = np.asarray(X)
        self._X_reduced = self._pca.transform(X_arr)
        n_cols = self._X_reduced.shape[1]
        self._component_names = [f"component_{i + 1}" for i in range(n_cols)]
        return self._X_reduced

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
            if X is None or y is None:
                raise ValueError(
                    "Dataset chưa có features/target. Hãy gọi set_target(target_column) trước khi preprocessing."
                )

            X_reduced = self.fit_transform(X)
            reduced_df = pd.DataFrame(
                X_reduced,
                columns=self._component_names,
                index=getattr(X, "index", None),
            )

            obj._features = reduced_df

            target_col = getattr(obj, "_target_column", None)
            if target_col:
                y_series = y if isinstance(y, pd.Series) else pd.Series(y, name=target_col)
                obj.data = pd.concat([reduced_df, y_series.rename(target_col)], axis=1)
                obj.set_target(target_col)

            self._status = "Success"
        except Exception as e:
            self._status = "Failed"
            self._error_message = str(e)

    def run(self, obj):
        self.visitTableDataset(obj)

    def log(self):
        print("\n" + "=" * 55)
        print(f"Step      : {self._step_name}")
        print(f"Dataset   : {self._dataset_path}")
        print(f"Status    : {self._status}")
        if self._status == "Success" and self._pca is not None:
            total_var = float(np.sum(self._pca.explained_variance_ratio_))
            print(f"Components: {self._pca.n_components_}")
            print(f"Variance  : {total_var:.4f} ({total_var * 100:.2f}%)")
        else:
            print(f"Error     : {self._error_message}")
        print("=" * 55)

    def visualize_umap(self, labels, class_names=None, title_suffix: str = "", n_samples: int = 1000):
        if self._X_reduced is None:
            raise ValueError("Chưa có dữ liệu để trực quan. Hãy gọi fit_transform hoặc run trước.")
        plot_dim_reduction_2d(
            self._X_reduced,
            labels=labels,
            class_names=class_names,
            method="umap",
            title_suffix=title_suffix,
            n_samples=n_samples,
        )

    def visualize_tsne(self, labels, class_names=None, title_suffix: str = "", n_samples: int = 1000):
        if self._X_reduced is None:
            raise ValueError("Chưa có dữ liệu để trực quan. Hãy gọi fit_transform hoặc run trước.")
        plot_dim_reduction_2d(
            self._X_reduced,
            labels=labels,
            class_names=class_names,
            method="tsne",
            title_suffix=title_suffix,
            n_samples=n_samples,
        )

def evaluate_feature_subset(X, y, cv=5):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=skf, scoring="f1_macro")
    return float(scores.mean())


def apply_statistical_filter(X, y, k, method="anova"):
    method_lower = method.lower()
    if method_lower == "anova":
        selector = ANOVAFiltering(k=k)
    elif method_lower == "chi2":
        selector = ChiSquareFiltering(k=k)
    elif method_lower in ("mutual_info", "mi"):
        selector = MutualInformationFiltering(k=k)
    else:
        raise ValueError(f"Phương pháp '{method}' không hợp lệ. Chọn 'anova', 'chi2', 'mutual_info'.")

    X_selected = selector.fit_transform(X, y)
    return X_selected, selector.selected_indices, selector.scores


def apply_model_based_filter(X, y, k, method="rf"):
    method_lower = method.lower()
    if method_lower == "rf":
        selector = RandomForestFiltering(k=k)
    elif method_lower == "gb":
        selector = GradientBoostingFiltering(k=k)
    else:
        raise ValueError(f"Phương pháp '{method}' không hợp lệ. Chọn 'rf' hoặc 'gb'.")

    X_selected = selector.fit_transform(X, y)
    return X_selected, selector.selected_indices, selector.importances


def apply_rfecv(X, y, cv=5):
    estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=skf,
        scoring="f1_macro",
        min_features_to_select=1,
        n_jobs=-1,
    )
    rfecv.fit(X, y)

    selected_indices = np.where(rfecv.support_)[0]
    X_selected = rfecv.transform(X)

    print(f"[RFECV] Số feature tối ưu: {rfecv.n_features_}")
    print(f"[RFECV] Các feature được chọn (index): {selected_indices}")

    return X_selected, selected_indices, rfecv


def apply_pca_reduction(X, n_components=0.95):
    reducer = DimensionReduction(n_components=n_components)
    X_pca = reducer.fit_transform(X)
    total_var = float(np.sum(reducer.pca.explained_variance_ratio_)) if reducer.pca else 0.0
    print(f"[PCA] Số thành phần giữ lại: {reducer.pca.n_components_ if reducer.pca else 'N/A'}")
    print(f"[PCA] Tổng phương sai giải thích: {total_var:.4f} ({total_var * 100:.2f}%)")
    return X_pca, reducer.pca
