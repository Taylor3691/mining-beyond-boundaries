"""
5 phương pháp / giảm chiều đặc trưng:
  1. evaluate_feature_subset  – Đánh giá F1-macro (RandomForest baseline)
  2. apply_statistical_filter – Lọc thống kê (ANOVA / Chi2 / MI)
  3. apply_model_based_filter – Lọc mô hình (RF / GB importance)
  4. apply_rfecv              – Tìm số feature tối ưu bằng RFECV
  5. apply_pca_reduction      – Giảm chiều PCA
"""

import numpy as np
import pandas as pd
from functools import partial

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2, mutual_info_classif,
    RFECV
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA

# 1. Evaluate – Đánh giá tập đặc trưng con
def evaluate_feature_subset(X, y, cv=5):
    """
    Đánh giá tập đặc trưng (X) bằng RandomForest với F1-macro trung bình.

    Parameters
    ----------
    X : numpy.ndarray hoặc pd.DataFrame
        Ma trận đặc trưng (n_samples, n_features).
    y : array-like
        Nhãn phân loại.
    cv : int
        Số fold cross-validation.

    Returns
    -------
    float
        Điểm F1-macro trung bình qua `cv` fold.
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=skf, scoring='f1_macro')
    return scores.mean()


# 2. Statistical Filter – Lọc thống kê
def apply_statistical_filter(X, y, k, method='anova'):
    """
    Chọn k đặc trưng tốt nhất bằng bộ lọc thống kê.

    Parameters
    ----------
    X : numpy.ndarray hoặc pd.DataFrame
        Ma trận đặc trưng.
    y : array-like
        Nhãn phân loại.
    k : int
        Số lượng đặc trưng cần giữ lại.
    method : str
        Phương pháp lọc: 'anova', 'chi2', 'mutual_info'.

    Returns
    -------
    X_selected : numpy.ndarray
        Ma trận đặc trưng đã lọc.
    selected_indices : numpy.ndarray
        Chỉ số của các đặc trưng được chọn.
    scores : numpy.ndarray
        Điểm score của bộ lọc cho từng đặc trưng.
    """
    method_lower = method.lower()

    if method_lower == 'anova':
        score_func = f_classif
    elif method_lower == 'chi2':
        score_func = chi2
    elif method_lower in ('mutual_info', 'mi'):
        score_func = partial(mutual_info_classif, random_state=42)
    else:
        raise ValueError(f"Phương pháp '{method}' không hợp lệ. Chọn 'anova', 'chi2', 'mutual_info'.")

    selector = SelectKBest(score_func=score_func, k=k)
    X_selected = selector.fit_transform(X, y)

    selected_indices = selector.get_support(indices=True)
    scores = selector.scores_

    return X_selected, selected_indices, scores


# 3. Model-based Filter – Lọc dựa trên mô hình
def apply_model_based_filter(X, y, k, method='rf'):
    """
    Chọn k đặc trưng quan trọng nhất dựa trên feature importance của mô hình.

    Parameters
    ----------
    X : numpy.ndarray hoặc pd.DataFrame
        Ma trận đặc trưng.
    y : array-like
        Nhãn phân loại.
    k : int
        Số lượng đặc trưng cần giữ lại.
    method : str
        Phương pháp: 'rf' (RandomForest), 'gb' (GradientBoosting).

    Returns
    -------
    X_selected : numpy.ndarray
        Ma trận đặc trưng đã lọc.
    selected_indices : numpy.ndarray
        Chỉ số của các đặc trưng được chọn (sắp xếp theo importance giảm dần).
    importances : numpy.ndarray
        Mảng feature importance cho từng đặc trưng gốc.
    """
    method_lower = method.lower()

    if method_lower == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif method_lower == 'gb':
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Phương pháp '{method}' không hợp lệ. Chọn 'rf' hoặc 'gb'.")

    model.fit(X, y)
    importances = model.feature_importances_

    # Lấy top-k đặc trưng quan trọng nhất
    selected_indices = np.argsort(importances)[::-1][:k]
    selected_indices_sorted = np.sort(selected_indices)  # Giữ thứ tự cột gốc

    X_arr = np.array(X)
    X_selected = X_arr[:, selected_indices_sorted]

    return X_selected, selected_indices_sorted, importances


# 4. RFECV – Recursive Feature Elimination with CV
def apply_rfecv(X, y, cv=5):
    """
    Tìm số lượng đặc trưng tối ưu bằng RFECV.

    Parameters
    ----------
    X : numpy.ndarray hoặc pd.DataFrame
        Ma trận đặc trưng.
    y : array-like
        Nhãn phân loại.
    cv : int
        Số fold cross-validation.

    Returns
    -------
    X_selected : numpy.ndarray
        Ma trận đặc trưng tối ưu.
    selected_indices : numpy.ndarray
        Chỉ số của các đặc trưng được chọn.
    rfecv : RFECV
        Object RFECV đã fit (chứa cv_results_, n_features_, ranking_).
    """
    estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=skf,
        scoring='f1_macro',
        min_features_to_select=1,
        n_jobs=-1
    )
    rfecv.fit(X, y)

    selected_indices = np.where(rfecv.support_)[0]
    X_selected = rfecv.transform(X)

    print(f"[RFECV] Số feature tối ưu: {rfecv.n_features_}")
    print(f"[RFECV] Các feature được chọn (index): {selected_indices}")

    return X_selected, selected_indices, rfecv


# 5. PCA Reduction – Giảm chiều PCA
def apply_pca_reduction(X, n_components=0.95):
    """
    Giảm chiều dữ liệu bằng PCA.

    Parameters
    ----------
    X : numpy.ndarray hoặc pd.DataFrame
        Ma trận đặc trưng.
    n_components : float hoặc int
        - float (0 < n_components < 1): Tỉ lệ phương sai cần giữ lại.
        - int: Số chiều đầu ra cố định.

    Returns
    -------
    X_pca : numpy.ndarray
        Ma trận đặc trưng sau PCA.
    pca : PCA
        Object PCA đã fit (chứa explained_variance_ratio_, components_).
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    total_var = np.sum(pca.explained_variance_ratio_)
    print(f"[PCA] Số thành phần giữ lại: {pca.n_components_}")
    print(f"[PCA] Tổng phương sai giải thích: {total_var:.4f} ({total_var*100:.2f}%)")

    return X_pca, pca
