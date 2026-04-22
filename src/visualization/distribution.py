import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from visualization.relationship import plot_dim_reduction_2d
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_class_distribution(classes: list, counts: list):
    """
    Vẽ biểu đồ cột so sánh số lượng mẫu giữa các lớp và tô nổi bật lớp nhiều/ít nhất.

    Input:
        classes: Danh sách tên các lớp dữ liệu.
        counts: Danh sách số lượng mẫu tương ứng với từng lớp.

    Output:
        None (hiển thị biểu đồ và lưu ảnh PNG).
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
    
    bar_colors = []
    for count in counts:
        if count == max_val:
            bar_colors.append('#ff4d4d')
        elif count == min_val:
            bar_colors.append('#4caf50')
        else:
            bar_colors.append('lightsteelblue')

    bars = plt.bar(classes, counts, color=bar_colors, edgecolor='black')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max_val * 0.02), 
                int(yval), ha='center', va='bottom', fontsize=10, fontweight='bold')

    if max_val > warning_threshold:
        plt.axhline(y=warning_threshold, color='orange', linestyle='--', linewidth=2, 
                    label=f'3x Minority Threshold ({warning_threshold} samples)')
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

def plot_histogram(pixel_data: np.ndarray, title_suffix: str = ""):
    """
    Vẽ Histogram phân phối cường độ pixel cho 3 kênh màu R-G-B.

    Input:
        pixel_data: Mảng dữ liệu pixel có dạng (N, 3) cho ba kênh màu.
        title_suffix: Hậu tố bổ sung vào tiêu đề biểu đồ.

    Output:
        None (hiển thị biểu đồ matplotlib).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(f'Pixel Intensity Distribution (Histogram) {title_suffix}', fontweight='bold', fontsize=16)
    for i, col in enumerate(['red', 'green', 'blue']):
        axes[i].hist(pixel_data[:, i], bins=100, color=col, alpha=0.7, density=True)
        axes[i].set_title(f"{col.capitalize()} Channel"); axes[i].grid(alpha=0.3)
    plt.tight_layout(); plt.show()

def plot_kde(pixel_data: np.ndarray, title_suffix: str = ""):
    """
    Vẽ đường mật độ KDE cho 3 kênh màu R-G-B và tự động lấy mẫu khi dữ liệu lớn.

    Input:
        pixel_data: Mảng dữ liệu pixel có dạng (N, 3) cho ba kênh màu.
        title_suffix: Hậu tố bổ sung vào tiêu đề biểu đồ.

    Output:
        None (hiển thị biểu đồ matplotlib).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(f'Kernel Density Estimation (KDE) {title_suffix}', fontweight='bold', fontsize=16)
    for i, col in enumerate(['red', 'green', 'blue']):
        data = pixel_data[:, i]
        if len(data) > 50000: data = np.random.choice(data, 50000)
        sns.kdeplot(data, color=col, fill=True, alpha=0.3, ax=axes[i], warn_singular=False)
        axes[i].set_title(f"{col.capitalize()} Channel"); axes[i].grid(alpha=0.3)
    plt.tight_layout(); plt.show()

def plot_distribution_by_class(images, labels, class_names, title_suffix=""):
    """
    Vẽ KDE phân phối màu theo từng lớp để so sánh sự khác biệt giữa các class.

    Input:
        images: Tập ảnh đầu vào có dạng (N, H, W, C).
        labels: Nhãn lớp của từng ảnh.
        class_names: Danh sách tên lớp tương ứng với nhãn số.
        title_suffix: Hậu tố bổ sung vào tiêu đề biểu đồ.

    Output:
        None (hiển thị biểu đồ matplotlib).
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    fig.suptitle(f'Color Distribution Comparison by Class (KDE) {title_suffix}', fontsize=16, fontweight='bold')
    
    channels = ['Red Channel', 'Green Channel', 'Blue Channel']
    labels_arr = np.array(labels)
    
    cmap = plt.get_cmap('tab10')

    for i in range(3):
        for class_idx, class_name in enumerate(class_names):
            class_mask = (labels_arr == class_idx)
            class_images = images[class_mask]
            
            if len(class_images) > 0:
                channel_data = class_images[:, :, :, i].flatten()
                if len(channel_data) > 50000:
                    channel_data = np.random.choice(channel_data, 50000, replace=False)
                
                sns.kdeplot(channel_data, ax=axes[i], label=class_name, linewidth=1.5, color=cmap(class_idx % 10))
        
        axes[i].set_title(channels[i], fontsize=13)
        axes[i].set_xlabel('Giá trị Pixel (0 - 255)', fontsize=12)
        axes[i].grid(axis='y', alpha=0.3, linestyle='--')
        
        if i == 1:
            axes[i].legend(title='Classes', bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=len(class_names)//2 + 1)

    axes[0].set_ylabel('Mật độ phân phối', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.show()
    
def plot_tsne(images, labels, class_names, title_suffix="", n_samples=1000):
    """
    Vẽ biểu đồ t-SNE để visualize sự phân bố dữ liệu trong không gian 2D.

    Input:
        images: Tập ảnh đầu vào có dạng (N, H, W, 3).
        labels: Nhãn lớp của từng ảnh.
        class_names: Danh sách tên lớp tương ứng với nhãn số.
        title_suffix: Hậu tố thêm vào tiêu đề biểu đồ.
        n_samples: Số mẫu tối đa dùng để chạy t-SNE.

    Output:
        None (hiển thị biểu đồ matplotlib).
    """
    images_arr = np.array(images)
    labels_arr = np.array(labels)
    
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

def plot_column_distribution(data_series: np.ndarray, column_name: str, test_name: str = ""):
    """
    Trực quan hóa phân phối của một cột dữ liệu số bằng Histogram kết hợp đường KDE.

    Input:
        data_series: Chuỗi dữ liệu số cần trực quan hóa phân phối.
        column_name: Tên cột dữ liệu để hiển thị trên biểu đồ.
        test_name: Tên kiểm định liên quan để ghi chú trên tiêu đề.

    Output:
        None (hiển thị biểu đồ matplotlib).
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

def plot_ks_test_results(orig_data, norm_data, method_name, d_stat, p_value):
    """
    Vẽ biểu đồ so sánh kết quả kiểm định Kolmogorov-Smirnov (K-S Test).
    Hiển thị 3 biểu đồ đường ECDF: Ảnh gốc, Ảnh chuẩn hóa và Overlay (đã chuẩn hóa z-score).

    Input:
        orig_data: Dữ liệu pixel gốc trước chuẩn hóa.
        norm_data: Dữ liệu pixel sau chuẩn hóa.
        method_name: Tên phương pháp chuẩn hóa đang đánh giá.
        d_stat: Giá trị thống kê D từ kiểm định K-S.
        p_value: Giá trị p-value từ kiểm định K-S.

    Output:
        None (hiển thị biểu đồ matplotlib).
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    colors, labs = ["#2980b9", "#c0392b"], ["Original", f"Normalized ({method_name})"]
    
    for i, (data, col, label) in enumerate(zip([orig_data, norm_data], colors, labs)):
        sns.ecdfplot(data, color=col, label=label, ax=axes[i], lw=2)
        axes[i].set_title(f"ECDF — {label}"); axes[i].grid(ls='--', alpha=0.5); axes[i].legend()

    def _z(arr):
        """
        Chuẩn hóa z-score cho mảng dữ liệu 1 chiều.

        Input:
            arr: Mảng dữ liệu số cần chuẩn hóa.

        Output:
            Giá trị trả về của hàm.
        """
        return (arr - arr.mean()) / (arr.std() + 1e-10)
    oz, nz = _z(orig_data), _z(norm_data)
    sns.ecdfplot(oz, color=colors[0], label="Original (z-scaled)", ax=axes[2], lw=2)
    sns.ecdfplot(nz, color=colors[1], label="Normalized (z-scaled)", ax=axes[2], lw=2, ls='--')
    
    # D-stat annotation
    so = np.sort(oz)
    eo, en = np.arange(1, len(so)+1)/len(so), np.searchsorted(np.sort(nz), so, side='right')/len(nz)
    idx = np.argmax(np.abs(eo - en))
    axes[2].annotate('', xy=(so[idx], en[idx]), xytext=(so[idx], eo[idx]), arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
    axes[2].set_title("ECDF Overlay (z-scaled)"); axes[2].grid(ls='--', alpha=0.5); axes[2].legend()

    alpha = 0.05
    res = "Chấp nhận H₀: Tương đồng" if p_value > alpha else "Bác bỏ H₀: Khác biệt"
    clr = '#1a7a1a' if p_value > alpha else '#b30000'
    fig.text(0.5, -0.05, f"K-S Stat D = {d_stat:.4f}  |  p = {p_value:.4e}  |  {res}", 
             ha='center', fontsize=12, color=clr, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor=clr))
    
    plt.suptitle(f"K-S Test Summary — Method: {method_name}", fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout(); plt.show()
 
def plot_ks_comparison_summary(ks_results: list, alpha: float = 0.05):
    """
    Vẽ biểu đồ cột so sánh tổng hợp các trị số D và p-value của nhiều phương pháp chuẩn hóa.
    Làm nổi bật các phương pháp bác bỏ hoặc chấp nhận giả thuyết H0 dựa trên ngưỡng alpha.

    Input:
        ks_results: Danh sách kết quả K-S, mỗi phần tử chứa Method, D và p-value.
        alpha: Mức ý nghĩa thống kê để quyết định chấp nhận/bác bỏ giả thuyết H0.

    Output:
        None (hiển thị biểu đồ matplotlib).
    """
    methods  = [r["Method"] for r in ks_results]
    d_stats  = [float(r["K-S Stat (D)"]) for r in ks_results]
    p_values = [max(float(r["p-value"]), 1e-300) for r in ks_results]
 
    colors_accept = '#4CAF50'
    colors_reject = '#f44336'
    bar_colors = [colors_accept if p > alpha else colors_reject for p in p_values]
 
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
 
    bars_d = axes[0].bar(methods, d_stats, color=bar_colors,
                         edgecolor='black', alpha=0.85)
    axes[0].set_title("So sánh K-S Statistic (D)", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Phương pháp chuẩn hóa", fontsize=12)
    axes[0].set_ylabel("K-S Statistic D  ∈  [0, 1]", fontsize=12)
    axes[0].set_ylim(0, max(d_stats) * 1.45 if max(d_stats) > 0 else 0.01)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
 
    for bar, d_val in zip(bars_d, d_stats):
        bar_h = bar.get_height()
        y_pos = bar_h + max(d_stats) * 0.02
        va    = 'bottom'
        if bar_h < max(d_stats) * 0.05:
            y_pos = bar_h / 2 if bar_h > 0 else 0.001
            va    = 'center'
        axes[0].text(bar.get_x() + bar.get_width() / 2, y_pos,
                     f'{d_val:.6f}', ha='center', va=va,
                     fontsize=10, fontweight='bold',
                     color='white' if va == 'center' else 'black')
 
    legend_elems = [
        mpatches.Patch(facecolor=colors_accept, edgecolor='black', label='Chấp nhận H₀ (p > α)'),
        mpatches.Patch(facecolor=colors_reject, edgecolor='black', label='Bác bỏ H₀ (p ≤ α)'),
    ]
    axes[0].legend(handles=legend_elems, loc='upper right', fontsize=10)
 
    p_max, p_min = max(p_values), min(p_values)
    use_log = (p_max / p_min > 100) if p_min > 0 else True
 
    if use_log:
        p_log    = [np.log10(p) for p in p_values]
        alpha_log = np.log10(alpha)
        bars_p = axes[1].bar(methods, p_log, color=bar_colors,
                             edgecolor='black', alpha=0.85)
        axes[1].axhline(y=alpha_log, color='orange', linestyle='--',
                        linewidth=2, label=f'Ngưỡng α = {alpha}  (log₁₀={alpha_log:.2f})')
        axes[1].set_ylabel("log₁₀(p-value)", fontsize=12)
 
        for bar, p_val, lv in zip(bars_p, p_values, p_log):
            label       = "Accept H₀" if p_val > alpha else "Reject H₀"
            label_color = '#2E7D32' if p_val > alpha else '#C62828'
            p_str       = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         lv + abs(min(p_log)) * 0.03 + 0.1,
                         f'{p_str}\n({label})',
                         ha='center', va='bottom', fontsize=9,
                         fontweight='bold', color=label_color)
        axes[1].set_title("So sánh p-value (log₁₀ scale)", fontsize=14, fontweight='bold')
    else:
        bars_p = axes[1].bar(methods, p_values, color=bar_colors,
                             edgecolor='black', alpha=0.85)
        axes[1].axhline(y=alpha, color='orange', linestyle='--',
                        linewidth=2, label=f'Ngưỡng α = {alpha}')
        axes[1].set_ylim(0, 1.1)
        axes[1].set_ylabel("p-value  ∈  [0, 1]", fontsize=12)
 
        for bar, p_val in zip(bars_p, p_values):
            label       = "Accept H₀" if p_val > alpha else "Reject H₀"
            label_color = '#2E7D32' if p_val > alpha else '#C62828'
            p_str       = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"
            y_pos = min(p_val + 0.03, 1.05)
            axes[1].text(bar.get_x() + bar.get_width() / 2, y_pos,
                         f'{p_str}\n({label})',
                         ha='center', va='bottom', fontsize=9,
                         fontweight='bold', color=label_color)
        axes[1].set_title("So sánh p-value (linear scale)", fontsize=14, fontweight='bold')
 
    axes[1].set_xlabel("Phương pháp chuẩn hóa", fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    axes[1].legend(fontsize=11)
 
    plt.suptitle(
        "Tổng hợp Kiểm định Kolmogorov-Smirnov: So sánh các Phương pháp Chuẩn hóa",
        fontsize=15, fontweight='bold', y=1.02
    )

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

    Input:
        before_df: DataFrame dữ liệu trước khi thực hiện scaling.
        after_dfs: Từ điển ánh xạ tên phương pháp sang DataFrame sau scaling.
        columns: Danh sách cột cần vẽ; nếu None sẽ tự chọn toàn bộ cột số.
        sample_size: Số mẫu tối đa của mỗi nhóm để giảm tải khi vẽ.
        ncols: Số cột subplot hiển thị trên mỗi hàng.
        title_suffix: Hậu tố hiển thị thêm ở tiêu đề tổng.

    Output:
        None (hiển thị biểu đồ matplotlib).
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

def plot_acf_chart(series, n_lags, title="ACF Plot"):
    """
    Vẽ biểu đồ Autocorrelation Function (ACF) cho chuỗi thời gian.

    Input:
        series: Chuỗi thời gian cần phân tích tự tương quan.
        n_lags: Số độ trễ tối đa cần hiển thị.
        title: Tiêu đề biểu đồ ACF.

    Output:
        None (hiển thị biểu đồ matplotlib).
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    plot_acf(series.dropna(), lags=n_lags, ax=ax, title=title, color='#1f77b4')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')
    plt.tight_layout()
    plt.show()

def plot_pacf_chart(series, n_lags, title="PACF Plot"):
    """
    Vẽ biểu đồ Partial Autocorrelation Function (PACF) cho chuỗi thời gian.

    Input:
        series: Chuỗi thời gian cần phân tích tự tương quan riêng phần.
        n_lags: Số độ trễ tối đa cần hiển thị.
        title: Tiêu đề biểu đồ PACF.

    Output:
        None (hiển thị biểu đồ matplotlib).
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    plot_pacf(series.dropna(), lags=n_lags, ax=ax, title=title, color='#ff7f0e', method='ywm')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('Lags')
    plt.ylabel('Partial Autocorrelation')
    plt.tight_layout()
    plt.show()