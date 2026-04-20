import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict

def plot_deduplicate_comparison(initial_count, final_count):
    """
    Visualizes the number of images before and after the deduplication process using a bar chart.
    
    Args:
        initial_count (int): Original number of images in the dataset.
        final_count (int): Remaining number of images after processing.
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
    Displays two images side-by-side with their Hash strings and Hamming distance for experimental analysis.
    
    Args:
        img1, img2 (numpy.ndarray): The two images to compare (BGR format from OpenCV).
        hash1, hash2 (str): Corresponding pHash strings.
        distance (int): Calculated Hamming distance between the two hashes.
        title (str): Title for the visualization.
    """
    # Convert BGR to RGB for correct display in Matplotlib
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Set title color: Green if similar (<=10), Red if different (>10)
    status_color = 'green' if distance <= 10 else 'red'
    fig.suptitle(f"{title}\nHamming Distance: {distance}", 
                 fontsize=14, fontweight='bold', color=status_color)
    
    # Plot Image 1 (Original/Reference)
    axes[0].imshow(img1_rgb)
    axes[0].set_title(f"Original / Side A\nHash: {hash1}", fontsize=9, family='monospace')
    axes[0].axis('off')
    
    # Plot Image 2 (Transformed/Duplicate)
    axes[1].imshow(img2_rgb)
    axes[1].set_title(f"Transformed / Side B\nHash: {hash2}", fontsize=9, family='monospace')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_normalization_comparison(img_orig: np.ndarray, img_norm: np.ndarray, method_name: str):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    h, w = img_orig.shape[:2]
    cy, cx = h // 2, w // 2
    
    # histogram với heatmap dùng channel red
    patches = [img_orig[cy-5:cy+5, cx-5:cx+5, 0], img_norm[cy-5:cy+5, cx-5:cx+5, 0]]
    hist_data = [img_orig[:,:,0].flatten(), img_norm[:,:,0].flatten()]
    
    img_norm_disp = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
    images_disp = [img_orig, img_norm_disp]
    
    is_diverging = img_norm.min() < 0
    titles = ["Original", f"Normalized ({method_name})"]

    for i in range(2):
        # ảnh gốc
        axes[i, 0].imshow(images_disp[i])
        axes[i, 0].set_title(f"{titles[i]} Image")
        axes[i, 0].axis('off')

        # histogram
        sns.histplot(hist_data[i], bins=50, kde=(i==1), 
                     color='gray' if i==0 else 'blue', ax=axes[i, 1])
        axes[i, 1].set_title(f"{titles[i]} Distribution (Channel 0)")

        # heatmap
        sns.heatmap(patches[i], annot=True, fmt=".0f" if i==0 else ".3f", 
                    cmap="Blues" if i==0 else ("coolwarm" if is_diverging else "viridis"), 
                    center=0 if (i==1 and is_diverging) else None,
                    ax=axes[i, 2], cbar=True)
        axes[i, 2].set_title(f"{titles[i]} ROI (Center 10x10)")

    plt.suptitle(f"Method: {method_name}", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def plot_feature_selection_comparison(k_list, results_dict):
    """
    Vẽ biểu đồ đường so sánh F1-score của các phương pháp lọc đặc trưng.
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

# Yêu cầu: Hàm vẽ plot time và chấm điểm khả nghi của 1 PHƯƠNG PHÁP
def plot_anomalies_single_method(dataset, anomaly_mask: np.ndarray, method_name: str = "Unknown Method"):
    """
    Plots a Time Plot with anomalies for a specific method.
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

# Yêu cầu: Hàm vẽ và chấm điểm khả nghi của TẤT CẢ PHƯƠNG PHÁP
def plot_anomalies_all_methods(dataset, anomalies_dict: Dict[str, np.ndarray]):
    """
    Plots a Time Plot and simultaneously displays anomalies from multiple methods.
    anomalies_dict: Dictionary with method name as key and boolean mask array as value.
    Ex: {'Z-Score': mask1, 'Isolation Forest': mask2, 'STL': mask3}
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