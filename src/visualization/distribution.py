import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from visualization.relationship import plot_dim_reduction_2d
import pandas as pd
from sklearn.manifold import TSNE

# ==========================================
# Task 11: Kiểm tra Imbalance

def plot_class_distribution(classes: list, counts: list):
    """
    Hàm vẽ Bar Chart so sánh số lượng ảnh giữa các lớp:
    - Nhấn mạnh lớp nhiều nhất (Đỏ) và ít nhất (Xanh lá)
    - Vẽ đường cảnh báo nếu có lớp có số lượng vượt quá 3 lần lớp nhỏ nhất
    """
    if not classes or not counts:
        print("No data available to plot")
        return

    # Tìm giá trị lớn nhất và nhỏ nhất
    max_val = max(counts)
    min_val = min(counts)
    
    # Ngưỡng cảnh báo: 3 lần lớp nhỏ nhất
    warning_threshold = min_val * 3

    plt.figure(figsize=(12, 6))
    
    # Khởi tạo mảng màu cho từng cột dựa trên điều kiện
    bar_colors = []
    for count in counts:
        if count == max_val:
            bar_colors.append('#ff4d4d')  # Màu Đỏ (Nhấn mạnh lớp nhiều nhất)
        elif count == min_val:
            bar_colors.append('#4caf50')  # Màu Xanh lá (Nhấn mạnh lớp ít nhất)
        else:
            bar_colors.append('lightsteelblue') # Màu mặc định cho các lớp còn lại

    bars = plt.bar(classes, counts, color=bar_colors, edgecolor='black')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max_val * 0.02), 
                int(yval), ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Vẽ đường cảnh báo ngang (horizontal line) nếu Max > 3 * Min
    if max_val > warning_threshold:
        plt.axhline(y=warning_threshold, color='orange', linestyle='--', linewidth=2, 
                    label=f'3x Minority Threshold ({warning_threshold} samples)')
        # Hiển thị chú thích (Legend) cho đường cảnh báo
        plt.legend(loc='upper right', framealpha=0.9)

    plt.title("Image Class Distribution", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Classes", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
        
    plt.ylim(0, max_val * 1.15)
        
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Lưu file biểu đồ 
    output_file = "class_distribution_chart.png"
    plt.savefig(output_file, dpi=300)
    print(f"\n[Visualizer] Chart successfully saved at: {output_file}")

# ==========================================
# Task 8: Phân tích phân phối Pixel (Histogram & KDE)
# ==========================================
def plot_histogram(pixel_data, title_suffix=""):
    """
    Vẽ biểu đồ Histogram. 
    Sử dụng Subplots (1 hàng, 3 cột) để tách biệt 3 kênh R, G, B.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(f'Pixel Intensity Distribution (Histogram) {title_suffix}', fontsize=16, fontweight='bold')
    
    colors = ['red', 'green', 'blue']
    labels = ['Red Channel', 'Green Channel', 'Blue Channel']

    for i in range(3):
        axes[i].hist(pixel_data[:, i], bins=256, range=[0, 256], color=colors[i], alpha=0.75)
        axes[i].set_title(labels[i], fontsize=13, color=colors[i])
        axes[i].set_xlabel('Giá trị Pixel (0 - 255)', fontsize=12)
        axes[i].grid(axis='y', alpha=0.3, linestyle='--')
        
    axes[0].set_ylabel('Tần suất (Số lượng pixel)', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def plot_kde(pixel_data, title_suffix=""):
    """
    Vẽ biểu đồ KDE.
    Sử dụng Subplots (1 hàng, 3 cột) để vẽ 3 phân phối của 3 kênh màu.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(f'Kernel Density Estimation (KDE) {title_suffix}', fontsize=16, fontweight='bold')
    
    colors = ['red', 'green', 'blue']
    labels = ['Red', 'Green', 'Blue']
    
    for i in range(3):
        # Subsample để vẽ KDE nhanh hơn nếu dữ liệu quá lớn
        data_to_plot = pixel_data[:, i]
        if len(data_to_plot) > 100000:
            data_to_plot = np.random.choice(data_to_plot, 100000, replace=False)

        sns.kdeplot(data_to_plot, color=colors[i], fill=True, alpha=0.3, ax=axes[i], linewidth=2)
        axes[i].set_title(f"{labels[i]} Channel", fontsize=13, color=colors[i])
        axes[i].set_xlabel('Giá trị Pixel (0 - 255)', fontsize=12)
        axes[i].grid(axis='y', alpha=0.3, linestyle='--')
        
    axes[0].set_ylabel('Mật độ phân phối', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def plot_distribution_by_class(images, labels, class_names, title_suffix=""):
    """
    Vẽ KDE plot cho từng class để so sánh sự khác biệt phân phối màu sắc giữa các lớp.
    
    Args:
        images (numpy.ndarray): Tập ảnh dạng (N, H, W, 3).
        labels (numpy.ndarray/list): Nhãn của từng ảnh (dạng số nguyên từ 0 đến num_classes-1).
        class_names (list): Danh sách tên các lớp tương ứng.
        title_suffix (str): Hậu tố thêm vào tiêu đề.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    fig.suptitle(f'Color Distribution Comparison by Class (KDE) {title_suffix}', fontsize=16, fontweight='bold')
    
    channels = ['Red Channel', 'Green Channel', 'Blue Channel']
    labels_arr = np.array(labels)
    
    # Tạo dải màu (colormap) để phân biệt các class dễ dàng hơn
    cmap = plt.get_cmap('tab10')

    for i in range(3):
        for class_idx, class_name in enumerate(class_names):
            # Lọc các ảnh thuộc class hiện tại
            class_mask = (labels_arr == class_idx)
            class_images = images[class_mask]
            
            if len(class_images) > 0:
                # Trích xuất kênh màu thứ i và phẳng hóa (flatten)
                channel_data = class_images[:, :, :, i].flatten()
                
                # TỐI ƯU HIỆU NĂNG: Lấy mẫu ngẫu nhiên 50,000 pixel để vẽ KDE không bị treo máy
                if len(channel_data) > 50000:
                    channel_data = np.random.choice(channel_data, 50000, replace=False)
                
                sns.kdeplot(channel_data, ax=axes[i], label=class_name, linewidth=1.5, color=cmap(class_idx % 10))
        
        axes[i].set_title(channels[i], fontsize=13)
        axes[i].set_xlabel('Giá trị Pixel (0 - 255)', fontsize=12)
        axes[i].grid(axis='y', alpha=0.3, linestyle='--')
        
        # Chỉ hiển thị chú thích (legend) ở biểu đồ giữa cho gọn
        if i == 1:
            axes[i].legend(title='Classes', bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=len(class_names)//2 + 1)

    axes[0].set_ylabel('Mật độ phân phối', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.show()
    
def plot_tsne(images, labels, class_names, title_suffix="", n_samples=1000):
    """
    Vẽ biểu đồ t-SNE để visualize sự phân bố dữ liệu trong không gian 2D.
    Args:
        images (numpy.ndarray/list): Tập ảnh dạng (N, H, W, 3).
        labels (numpy.ndarray/list): Nhãn của từng ảnh (dạng số nguyên).
        class_names (list): Danh sách tên các lớp tương ứng.
        title_suffix (str): Hậu tố thêm vào tiêu đề.
        n_samples (int): Giới hạn số ảnh để t-SNE không bị chậm. 
    """
    images_arr = np.array(images)
    labels_arr = np.array(labels)
    
    # Subsampling nếu dữ liệu lớn hơn giới hạn
    if len(images_arr) > n_samples:
        indices = np.random.choice(len(images_arr), n_samples, replace=False)
        data_to_plot = images_arr[indices]
        labels_to_plot = labels_arr[indices]
    else:
        data_to_plot = images_arr
        labels_to_plot = labels_arr
        
    n_samples_actual = len(data_to_plot)
    if n_samples_actual == 0:
        print("No data available to plot t-SNE.")
        return
        
    # Flatten từng ảnh về 1D để chạy t-SNE
    flattened_data = data_to_plot.reshape(n_samples_actual, -1)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(1, n_samples_actual - 1)), max_iter=1000)
    tsne_results = tsne.fit_transform(flattened_data)
    
    df = pd.DataFrame()
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]
    df['Label'] = [class_names[lbl] for lbl in labels_to_plot]
    
    # Dùng fig/ax để tương thích với vòng lặp trong Jupyter
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="Label",
        palette=sns.color_palette("tab10", len(np.unique(labels_to_plot))),
        data=df,
        legend="full",
        alpha=0.7,
        ax=ax
    )
    
    ax.set_title(f't-SNE Mapping (n={n_samples_actual}) {title_suffix}', fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    
    from IPython.display import display
    display(fig)
    plt.close(fig)
    plt.close()
# ==========================================
# Task 10-Module 2: Phân tích phân phối dữ liệu (Histogram & KDE)
# ==========================================

def plot_column_distribution(data_series: np.ndarray, column_name: str, test_name: str = ""):
    """
    Trực quan hóa phân phối của một cột dữ liệu số bằng Histogram kết hợp đường KDE.
    """
    plt.figure(figsize=(10, 6))
    
    # Vẽ biểu đồ phân phối
    sns.histplot(data_series, kde=True, color='royalblue', bins=30, stat="density", linewidth=0)
    
    plt.title(f"Phân phối dữ liệu: {column_name} \n(Trước kiểm định {test_name})", fontsize=14, pad=15)
    plt.xlabel(f"Giá trị của {column_name}", fontsize=12)
    plt.ylabel("Mật độ (Density)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()


def plot_violin_comparison(
    before_df: pd.DataFrame,
    after_dfs: dict[str, pd.DataFrame],
    columns: list[str] | None = None,
    sample_size: int = 3000,
    ncols: int = 2,
    title_suffix: str = "",
):
    """
    Vẽ violin plot để so sánh phân phối trước/sau scaling theo từng cột.

    Args:
        before_df (pd.DataFrame): Dữ liệu trước scaling.
        after_dfs (dict[str, pd.DataFrame]): Từ điển {tên_phương_pháp: DataFrame sau scaling}.
        columns (list[str] | None): Danh sách cột cần vẽ. Nếu None, tự động lấy tất cả cột số.
        sample_size (int): Số mẫu tối đa cho mỗi nhóm để tránh vẽ quá nặng.
        ncols (int): Số cột subplot trên mỗi hàng.
        title_suffix (str): Hậu tố hiển thị thêm ở tiêu đề tổng.
    """
    if not isinstance(before_df, pd.DataFrame) or before_df.empty:
        print("[Violin] before_df không hợp lệ hoặc rỗng.")
        return

    if not after_dfs:
        print("[Violin] after_dfs đang rỗng, không có dữ liệu để so sánh.")
        return

    source_map = {"Trước scaling": before_df}
    source_map.update(after_dfs)

    if columns is None:
        columns = before_df.select_dtypes(include=[np.number]).columns.tolist()

    valid_columns = [
        col for col in columns
        if all(isinstance(df, pd.DataFrame) and col in df.columns for df in source_map.values())
    ]

    if not valid_columns:
        print("[Violin] Không tìm thấy cột hợp lệ để vẽ.")
        return

    ncols = max(1, int(ncols))
    n_plots = len(valid_columns)
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False)
    fig.suptitle(f"So sánh phân phối bằng Violin Plot {title_suffix}", fontsize=16, fontweight="bold")

    for idx, col in enumerate(valid_columns):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        long_parts = []
        for label, df in source_map.items():
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if sample_size and sample_size > 0 and len(series) > sample_size:
                series = series.sample(sample_size, random_state=42)

            long_parts.append(
                pd.DataFrame(
                    {
                        "Phiên bản": label,
                        "Giá trị": series.values,
                    }
                )
            )

        plot_df = pd.concat(long_parts, ignore_index=True)

        sns.violinplot(
            data=plot_df,
            x="Phiên bản",
            y="Giá trị",
            inner="quartile",
            cut=0,
            ax=ax,
        )
        ax.set_title(col, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("Giá trị")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Ẩn các ô subplot thừa
    for idx in range(n_plots, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
