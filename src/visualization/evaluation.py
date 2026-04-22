import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_color_space_evaluation(results: list, save_path: str = "color_space_eval.png"):
    """
    Vẽ biểu đồ kép so sánh phương sai giải thích và F1-Score giữa các không gian màu.

    Input:
        results: Danh sách kết quả đánh giá theo từng không gian màu.
        save_path: Đường dẫn lưu ảnh biểu đồ sau khi vẽ.

    Output:
        None (hiển thị và lưu biểu đồ matplotlib).
    """
    if not results:
        print("[Visualizer Error] Danh sách kết quả rỗng.")
        return

    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Đánh giá Hiệu quả Tiền xử lý Không gian màu", fontsize=16, fontweight='bold')

    # Biểu đồ 1: Phương sai giải thích
    sns.barplot(data=df, x="Color_Space", y="Variance_Ratio_Sum", ax=axes[0], palette="viridis", edgecolor='black')
    axes[0].set_title("Khả năng giữ thông tin (Tỷ lệ Phương sai giải thích - PCA k=50)")
    axes[0].set_ylabel("Explained Variance Ratio Sum")
    axes[0].set_ylim(0, 1.0)
    for p in axes[0].patches:
        axes[0].annotate(format(p.get_height(), '.4f'), 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')

    # Biểu đồ 2: F1-Score
    sns.barplot(data=df, x="Color_Space", y="F1_Score", ax=axes[1], palette="magma", edgecolor='black')
    axes[1].set_title("Độ chính xác Mô hình (F1-Score trung bình)")
    axes[1].set_ylabel("F1-Score")
    for p in axes[1].patches:
        axes[1].annotate(format(p.get_height(), '.4f'), 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
def plot_normalization_learning_curves(results_dict: dict, save_path: str = "norm_learning_curves.png"):
    """
    Vẽ đường cong hội tụ (Learning Curves) so sánh độ mượt và tốc độ hội tụ của 4 phương pháp Chuẩn hóa.

    Input:
        results_dict: Từ điển ánh xạ tên phương pháp chuẩn hóa sang chuỗi độ chính xác theo epoch.
        save_path: Đường dẫn lưu ảnh biểu đồ sau khi vẽ.

    Output:
        None (hiển thị và lưu biểu đồ matplotlib).
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    
    # results_dict có dạng: {"minmax_01": [0.2, 0.4, 0.5...], "zscore_global": [0.3, 0.6...]}
    for method, curve in results_dict.items():
        plt.plot(range(1, len(curve)+1), curve, marker='o', linewidth=2, label=method)
    
    plt.title("Tốc độ Hội tụ và Sự ổn định của các phương pháp Chuẩn hóa", fontsize=14, fontweight='bold')
    plt.xlabel("Số Epochs", fontsize=12)
    plt.ylabel("Độ chính xác (Accuracy) trên Test set", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Phương pháp Chuẩn hóa")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

def plot_scree_pca(explained_variance_ratio: np.ndarray, save_path: str = "scree_plot_pca.png"):
    """
    Vẽ biểu đồ Scree Plot (Cumulative Explained Variance) để tìm mốc 90%, 95%, 99%.

    Input:
        explained_variance_ratio: Mảng tỷ lệ phương sai giải thích của từng thành phần PCA.
        save_path: Đường dẫn lưu ảnh biểu đồ sau khi vẽ.

    Output:
        None (hiển thị và lưu biểu đồ matplotlib).
    """
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='.', linestyle='-', color='#1f77b4')
    
    # Kẻ các đường ngưỡng phương sai
    thresholds = [0.90, 0.95, 0.99]
    colors = ['#d62728', '#2ca02c', '#ff7f0e']
    
    for threshold, color in zip(thresholds, colors):
        # Tìm số k components đầu tiên vượt ngưỡng
        indices = np.where(cumulative_variance >= threshold)[0]
        if len(indices) > 0:
            k = indices[0] + 1
            plt.axhline(y=threshold, color=color, linestyle='--', alpha=0.8, label=f'{int(threshold*100)}% Variance')
            plt.axvline(x=k, color=color, linestyle=':', alpha=0.8)
            plt.scatter(k, threshold, color=color, zorder=5)
            plt.text(k + 2, threshold - 0.03, f'k={k}', color=color, fontweight='bold')
            
    plt.title('Scree Plot: Mức độ giải thích phương sai (PCA)', fontsize=14, fontweight='bold')
    plt.xlabel('Số lượng Principal Components (k)', fontsize=12)
    plt.ylabel('Cumulative Explained Variance', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

def plot_pca_scatter_2d(X_pca_2d: np.ndarray, labels: np.ndarray, class_names: list, save_path: str = "pca_scatter_2d.png"):
    """
    Trực quan hóa dữ liệu trên không gian 2D với 2 Principal Components đầu tiên.

    Input:
        X_pca_2d: Ma trận đặc trưng sau khi giảm chiều PCA xuống 2 thành phần.
        labels: Mảng nhãn lớp tương ứng với từng mẫu dữ liệu.
        class_names: Danh sách tên lớp để hiển thị trên chú giải.
        save_path: Đường dẫn lưu ảnh biểu đồ sau khi vẽ.

    Output:
        None (hiển thị và lưu biểu đồ matplotlib).
    """
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    palette = sns.color_palette("tab10", len(unique_labels))
    
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        plt.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
                    color=palette[i], label=class_names[lbl], alpha=0.7, edgecolors='w', s=50)
        
    plt.title("Phân bố dữ liệu trên 2 Principal Components đầu tiên", fontsize=14, fontweight='bold')
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

def plot_pca_scatter_3d(X_pca_3d: np.ndarray, labels: np.ndarray, class_names: list, save_path: str = "pca_scatter_3d.png"):
    """
    Trực quan hóa dữ liệu trên không gian 3D với 3 Principal Components đầu tiên.

    Input:
        X_pca_3d: Ma trận đặc trưng sau khi giảm chiều PCA xuống 3 thành phần.
        labels: Mảng nhãn lớp tương ứng với từng mẫu dữ liệu.
        class_names: Danh sách tên lớp để hiển thị trên chú giải.
        save_path: Đường dẫn lưu ảnh biểu đồ sau khi vẽ.

    Output:
        None (hiển thị và lưu biểu đồ matplotlib).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = np.unique(labels)
    palette = sns.color_palette("tab10", len(unique_labels))
    
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(X_pca_3d[mask, 0], X_pca_3d[mask, 1], X_pca_3d[mask, 2],
                   color=palette[i], label=class_names[lbl], alpha=0.7, s=40)
        
    ax.set_title("Phân bố dữ liệu 3D (PC1, PC2, PC3)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()