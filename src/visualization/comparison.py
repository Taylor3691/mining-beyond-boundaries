import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict

def plot_deduplicate_comparison(initial_count, final_count):
    """
    Trực quan hóa số lượng ảnh trước và sau khi loại bỏ trùng lặp bằng biểu đồ thanh.
    
    Input:
        initial_count: Số lượng ảnh ban đầu trước khi lọc trùng lặp.
        final_count: Số lượng ảnh còn lại sau khi xử lý loại bỏ trùng lặp.

    Output:
        None (hiển thị biểu đồ matplotlib).
    """
    labels = ['Before Removal', 'After Removal']
    counts = [initial_count, final_count]

    plt.figure(figsize=(8, 6))
    # Standard color scheme: Blue for Before, Orange for After
    bars = plt.bar(labels, counts, color=['#1f77b4', '#ff7f0e'])
    
    plt.ylabel('Number of Images')
    plt.title('Image Count Before and After Deduplication')
    
    # Display the exact count on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), 
                 va='bottom', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_phash_comparison(img1, img2, hash1, hash2, distance, title="pHash Comparison"):
    """
    Hiển thị hai ảnh cạnh nhau kèm chuỗi Hash và khoảng cách Hamming.
    
    Input:
        img1: Ảnh thứ nhất dùng làm mẫu tham chiếu.
        img2: Ảnh thứ hai cần đối chiếu với ảnh tham chiếu.
        hash1: Chuỗi pHash tương ứng với ảnh thứ nhất.
        hash2: Chuỗi pHash tương ứng với ảnh thứ hai.
        distance: Khoảng cách Hamming giữa hai chuỗi hash.
        title: Tiêu đề chính của biểu đồ.

    Output:
        None (hiển thị biểu đồ matplotlib).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Set title color: Green if similar (<=10), Red if different (>10)
    status_color = 'green' if distance <= 10 else 'red'
    fig.suptitle(f"{title}\nHamming Distance: {distance}", 
                 fontsize=14, fontweight='bold', color=status_color)
    
    # Plot Image 1 (Original/Reference)
    axes[0].imshow(img1)
    axes[0].set_title(f"Original / Side A\nHash: {hash1}", fontsize=9, family='monospace')
    axes[0].axis('off')
    
    # Plot Image 2 (Transformed/Duplicate)
    axes[1].imshow(img2)
    axes[1].set_title(f"Transformed / Side B\nHash: {hash2}", fontsize=9, family='monospace')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_normalization_comparison(img_orig: np.ndarray, img_norm: np.ndarray, method: str):
    """
    Vẽ biểu đồ so sánh ảnh gốc và ảnh sau chuẩn hóa, kèm Histogram và KDE theo từng kênh RGB.

    Input:
        img_orig: Mảng ảnh gốc trước chuẩn hóa.
        img_norm: Mảng ảnh sau khi áp dụng chuẩn hóa.
        method: Tên phương pháp chuẩn hóa để hiển thị trên tiêu đề.

    Output:
        None (hiển thị biểu đồ matplotlib).
    """
    # Xử lý hiển thị: Ảnh gốc giữ nguyên, ảnh chuẩn hóa được chuẩn hóa hiển thị về [0, 255]
    orig_rgb = img_orig
    norm_disp = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
    norm_rgb = (norm_disp * 255).astype(np.uint8) if img_norm.ndim == 3 else norm_disp

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    titles = ["Original", f"Normalized ({method})"]
    imgs, data = [orig_rgb, norm_rgb], [img_orig, img_norm]
    colors = ['red', 'green', 'blue']

    for r in range(2):
        # Cột 0: Hiển thị hình ảnh
        axes[r, 0].imshow(imgs[r])
        axes[r, 0].set_title(titles[r])
        axes[r, 0].axis('off')
        
        # Cột 1-3: Hiển thị phân phối pixel từng kênh R, G, B
        for c in range(3):
            ax = axes[r, c+1]
            ch_data = data[r][:, :, c].flatten()
            
            # Vẽ Histogram và đường KDE chồng lên nhau
            ax.hist(ch_data, bins=100, color=colors[c], alpha=0.6, density=True)
            sns.kdeplot(ch_data, ax=ax, color=colors[c], lw=1.5)
            
            ax.set_title(f"{colors[c].capitalize()} Channel")
            ax.grid(alpha=0.3)
            
    plt.suptitle(f"Normalization Comparison Analysis: {method}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_feature_selection_comparison(k_list, results_dict):
    """
    Vẽ biểu đồ đường so sánh F1-score của các phương pháp lọc đặc trưng.

    Input:
        k_list: Danh sách số lượng đặc trưng được chọn ở từng mốc đánh giá.
        results_dict: Từ điển ánh xạ tên phương pháp sang danh sách F1-score tương ứng.

    Output:
        None (hiển thị biểu đồ matplotlib).
    """
    # Palette chuyên dụng để phân biệt rõ các phương pháp
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X']
    colors = sns.color_palette("Set1", n_colors=len(results_dict))

    plt.figure(figsize=(12, 7))

    for idx, (method_name, scores) in enumerate(results_dict.items()):
        marker = markers[idx % len(markers)]
        plt.plot(
            k_list, scores,
            marker=marker,
            linewidth=2.5,
            markersize=8,
            label=method_name,
            color=colors[idx]
        )

    plt.title("Feature Selection: F1-Score Comparison", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Number of Selected Features (k)", fontsize=13)
    plt.ylabel("F1-Score (macro)", fontsize=13)
    plt.xticks(k_list)
    plt.legend(title="Method", fontsize=11, title_fontsize=12, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_time_series(dates, values, feature_name, region="World"):
    """
    Trực quan hóa chuỗi thời gian để phân tích Trend, Seasonality và Noise bằng mắt.

    Input:
        dates: Trục thời gian của chuỗi dữ liệu.
        values: Chuỗi giá trị quan sát theo thời gian.
        feature_name: Tên biến đang được trực quan hóa.
        region: Khu vực khảo sát (mặc định là World).

    Output:
        None (hiển thị biểu đồ matplotlib).
    """
    plt.figure(figsize=(15, 6))
    
    # 1. Vẽ dữ liệu gốc (Thường có nhiều Noise và Seasonality)
    plt.plot(dates, values, label='Daily Raw Data', color='#1f77b4', linewidth=1.5, alpha=0.6)
    
    # 2. Vẽ đường Trung bình trượt 7 ngày để ép phẳng nhiễu, làm lộ ra Trend
    rolling_mean = values.rolling(window=7, min_periods=1).mean()
    plt.plot(dates, rolling_mean, label='7-Day Moving Average (Trend focus)', color='red', linewidth=2.5)
    
    plt.title(f"Time Plot of {feature_name} in {region}", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(f"Number of {feature_name}", fontsize=12)
    
    # Format trục X cho dễ nhìn ngày tháng
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
def plot_rolling_statistics(dates, values, feature_name, stat_type="Mean", region="World"):
    """
    Vẽ 4 đường trên cùng 1 biểu đồ: Data gốc, Window 7, 30, 90 cho Mean hoặc STD.

    Input:
        dates: Trục thời gian của chuỗi dữ liệu.
        values: Chuỗi giá trị quan sát theo thời gian.
        feature_name: Tên biến cần phân tích thống kê trượt.
        stat_type: Loại thống kê trượt cần vẽ (Mean hoặc STD).
        region: Khu vực khảo sát để hiển thị trên tiêu đề.

    Output:
        None (hiển thị biểu đồ matplotlib).
    """
    plt.figure(figsize=(15, 7))
    
    # Vẽ dữ liệu gốc
    plt.plot(dates, values, label='Daily Raw Data', color='lightgray', alpha=0.8, linewidth=1.5)
    
    # Tính toán và vẽ các đường Rolling
    if stat_type.lower() == "mean":
        roll_7 = values.rolling(window=7, min_periods=1).mean()
        roll_30 = values.rolling(window=30, min_periods=1).mean()
        roll_90 = values.rolling(window=90, min_periods=1).mean()
        title_stat = "Rolling Mean"
    elif stat_type.lower() == "std":
        roll_7 = values.rolling(window=7, min_periods=1).std().fillna(0) # Tránh NaN ở những ngày đầu
        roll_30 = values.rolling(window=30, min_periods=1).std().fillna(0)
        roll_90 = values.rolling(window=90, min_periods=1).std().fillna(0)
        title_stat = "Rolling Standard Deviation (STD)"
    else:
        raise ValueError("[ERROR] stat_type chỉ nhận 'Mean' hoặc 'STD'")

    # Vẽ 3 đường Rolling với độ dày và màu sắc khác nhau
    plt.plot(dates, roll_7, label=f'7-Day {title_stat} (Short-term)', color='blue', linewidth=1.5)
    plt.plot(dates, roll_30, label=f'30-Day {title_stat} (Mid-term)', color='orange', linewidth=2.5)
    plt.plot(dates, roll_90, label=f'90-Day {title_stat} (Long-term)', color='red', linewidth=3.5)

    plt.title(f"{title_stat} of {feature_name} in {region}", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(f"Value ({stat_type})", fontsize=12)
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_anomalies_single_method(dataset, anomaly_mask: np.ndarray, method_name: str = "Unknown Method"):
    """
    Vẽ chuỗi thời gian và đánh dấu các điểm dị thường theo một phương pháp.

    Input:
        dataset: Đối tượng dữ liệu chuỗi thời gian đã được nạp và set target.
        anomaly_mask: Mảng boolean, True tại các vị trí được xem là dị thường.
        method_name: Tên phương pháp phát hiện dị thường.

    Output:
        None (hiển thị biểu đồ matplotlib).
    """
    if dataset.data is None or dataset.target is None:
        raise ValueError("Dataset is not loaded or target is not set.")

    time_col = dataset.data[dataset._time_column]
    values = dataset.target

    plt.figure(figsize=(16, 6))
    
    # Plot original data line
    plt.plot(time_col, values, label='Original Data (Target)', color='steelblue', alpha=0.7, linewidth=1.5)
    
    # Scatter anomaly points
    plt.scatter(time_col[anomaly_mask], values[anomaly_mask], 
                color='red', label=f'Anomalies ({method_name})', 
                s=50, zorder=5, edgecolors='black')

    plt.title(f'Anomaly Detection - Method: {method_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel(dataset._target_column, fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_anomalies_all_methods(dataset, anomalies_dict: Dict[str, np.ndarray]):
    """
    Vẽ chuỗi thời gian và hiển thị đồng thời các điểm dị thường từ nhiều phương pháp.

    Input:
        dataset: Đối tượng dữ liệu chuỗi thời gian đã được nạp và set target.
        anomalies_dict: Từ điển ánh xạ tên phương pháp sang mảng boolean đánh dấu dị thường.

    Output:
        None (hiển thị biểu đồ matplotlib).
    """
    if dataset.data is None or dataset.target is None:
        raise ValueError("Dataset is not loaded or target is not set.")

    time_col = dataset.data[dataset._time_column]
    values = dataset.target

    plt.figure(figsize=(16, 8))
    
    # Plot original data line
    plt.plot(time_col, values, label='Original Data', color='gray', alpha=0.5, linewidth=1.5)
    
    # Color palette for differentiating methods
    color_palette = ['red', 'orange', 'purple', 'green', 'magenta']
    
    # Scatter each method onto the plot
    for idx, (method_name, mask) in enumerate(anomalies_dict.items()):
        color = color_palette[idx % len(color_palette)]
        plt.scatter(time_col[mask], values[mask], 
                    color=color, label=f'{method_name} ({mask.sum()} pts)', 
                    s=60 - (idx*10), # Decrease size gradually to keep them visible if overlapped
                    zorder=5+idx, alpha=0.8, edgecolors='black')

    plt.title('Comparison of Anomalies Across Methods', fontsize=15, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel(dataset._target_column, fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()