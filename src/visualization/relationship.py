import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import FancyArrowPatch

def visualize_brightness_contrast_boxplot(df: pd.DataFrame) -> None:
    """
    Vẽ boxplot độ sáng và độ tương phản theo từng lớp.

    Parameters
    ----------
    df : pd.DataFrame
        Output của compute_brightness_contrast_per_class()
        có columns: ['class_name', 'brightness', 'contrast']
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Phân tích Độ sáng & Độ tương phản theo Lớp", fontsize=14, fontweight="bold")

    # --- Boxplot Brightness ---
    sns.boxplot(
        data=df, x="class_name", y="brightness",
        palette="Set2", ax=axes[0]
    )
    axes[0].set_title("Độ sáng (Mean Intensity) theo Lớp")
    axes[0].set_xlabel("Lớp")
    axes[0].set_ylabel("Mean Intensity [0–255]")
    axes[0].tick_params(axis="x", rotation=45)

    # --- Boxplot Contrast ---
    sns.boxplot(
        data=df, x="class_name", y="contrast",
        palette="Set1", ax=axes[1]
    )
    axes[1].set_title("Độ tương phản (Std Intensity) theo Lớp")
    axes[1].set_xlabel("Lớp")
    axes[1].set_ylabel("Std Intensity [0–127.5]")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("brightness_contrast_boxplot.png", dpi=150, bbox_inches="tight")
    plt.show()

#  Task 22: Vẽ đường cong SSIM, đầu vào nhận vào mảng các giá trị SSIM
# trung bình sau khi đã resize, sau đó vẽ line chart (Hoặc một đồ thị đường)
# cong gì đó theo ý thầy, check thử xem đồ thị đó có phải line chart rồi hãng vẽ
def plot_ssim_curve(sizes: list, ssim_scores: list, save_path: str = "ssim_vs_size_curve.png"):
    """
    Vẽ đồ thị đường (Line Chart) thể hiện mối liên hệ giữa kích thước và chỉ số SSIM 
    """
    if len(sizes) != len(ssim_scores) or len(sizes) == 0:
        print("[Visualizer Error] Sizes and SSIM scores lists must have the same length and cannot be empty.")
        return

    # Sắp xếp size để đường vẽ chạy từ trái sang phải (32 -> 64 -> 128)
    sorted_pairs = sorted(zip(sizes, ssim_scores))
    sorted_sizes = [pair[0] for pair in sorted_pairs]
    sorted_ssim = [pair[1] for pair in sorted_pairs]

    plt.figure(figsize=(9, 6))
    
    # Vẽ Line Chart
    plt.plot(sorted_sizes, sorted_ssim, marker='o', linestyle='-', color='royalblue', 
             linewidth=2.5, markersize=8, label='SSIM Trend')

    # Tính toán khoảng đệm trục Y động để ghi chú text không bị dính vào điểm vẽ
    y_min, y_max = min(sorted_ssim), max(sorted_ssim)
    y_range = y_max - y_min if y_max != y_min else 0.1
    text_offset = y_range * 0.05  # Nâng text lên 5% của tổng chiều cao trục Y

    # Ghi chú giá trị SSIM
    for x, y in zip(sorted_sizes, sorted_ssim):
        plt.text(x, y + text_offset, f"{y:.4f}", 
                 ha='center', va='bottom', fontsize=11, fontweight='bold', color='darkblue')

    plt.title("Relationship Between Image Resize Dimension and SSIM", fontsize=15, pad=15)
    plt.xlabel("Target Image Size (NxN Pixels)", fontsize=12)
    plt.ylabel("Average SSIM Score (0.0 - 1.0)", fontsize=12)
    
    # Hiển thị text trực quan hơn ở trục X 
    plt.xticks(sorted_sizes, labels=[f"{s}x{s}" for s in sorted_sizes])
    
    # Giới hạn trục Y
    padding = y_range * 0.15
    plt.ylim(y_min - padding, y_max + padding * 2)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Lưu biểu đồ
    plt.savefig(save_path, dpi=300)
    print(f"\n[Visualizer] SSIM relationship curve successfully saved at: {save_path}")

# =================================================================================
# TASK 1.1: BỔ SUNG CODE VẼ HEATMAP (PEARSON VÀ SPEARMAN) CHO DỮ LIỆU BẢNG
# =================================================================================

def plot_pearson_heatmap(df: pd.DataFrame, title="Pearson Correlation Heatmap"):
    plt.figure(figsize=(14, 12))
    # Tính ma trận tương quan Pearson
    corr = df.corr(method='pearson')
    
    # Tạo mặt nạ (mask) để che nửa tam giác trên (tránh trùng lặp)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Vẽ Heatmap (Dùng colormap coolwarm để dễ nhìn Tương quan Âm/Dương)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                vmin=-1, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_spearman_heatmap(df: pd.DataFrame, title="Spearman Correlation Heatmap"):
    plt.figure(figsize=(14, 12))
    # Tính ma trận tương quan Spearman
    corr = df.corr(method='spearman')
    
    # Tạo mặt nạ (mask) để che nửa tam giác trên
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Vẽ Heatmap (Dùng colormap viridis để phân biệt với Pearson)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='viridis',
                vmin=-1, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_dim_reduction_2d(X, labels, class_names=None, method='tsne',
                          title_suffix="", n_samples=1000):
    """
    Vẽ Scatter Plot 2D sau khi giảm chiều bằng t-SNE hoặc UMAP.
    """
    from sklearn.manifold import TSNE

    X = np.array(X)
    labels = np.array(labels)

    # --- Subsampling ---
    if len(X) > n_samples:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X = X[indices]
        labels = labels[indices]

    n_actual = len(X)
    if n_actual == 0:
        print(f"[plot_dim_reduction_2d] No data available to plot ({method}).")
        return

    # --- Tính toán giảm chiều ---
    method_lower = method.lower()
    if method_lower == 'tsne':
        reducer = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, max(1, n_actual - 1)),
            max_iter=1000
        )
        embedding = reducer.fit_transform(X)
        method_label = "t-SNE"
    elif method_lower == 'umap':
        try:
            import umap
        except ImportError:
            print("[plot_dim_reduction_2d] UMAP chưa được cài đặt. Chạy: pip install umap-learn")
            return
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(X)
        method_label = "UMAP"
    else:
        raise ValueError(f"Phương pháp '{method}' không hợp lệ. Chọn 'tsne' hoặc 'umap'.")

    # --- Chuẩn bị DataFrame ---
    if class_names is not None:
        label_names = [class_names[int(lbl)] for lbl in labels]
    else:
        label_names = [str(lbl) for lbl in labels]

    df = pd.DataFrame({
        'Dim-1': embedding[:, 0],
        'Dim-2': embedding[:, 1],
        'Label': label_names
    })

    # --- Vẽ Scatter Plot (giữ nguyên thiết kế gốc) ---
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        x='Dim-1', y='Dim-2',
        hue='Label',
        palette=sns.color_palette("tab10", len(np.unique(labels))),
        data=df,
        legend="full",
        alpha=0.7,
        ax=ax
    )

    ax.set_title(f'{method_label} Mapping (n={n_actual}) {title_suffix}',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel(f'{method_label} Component 1', fontsize=12)
    ax.set_ylabel(f'{method_label} Component 2', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()

    from IPython.display import display
    display(fig)
    plt.close(fig)


def plot_granger_causality_directed_graph(
    p_value_matrix: pd.DataFrame,
    alpha: float = 0.05,
    title: str = "Granger Causality Directed Graph",
    save_path: str | None = None,
    show_edge_labels: bool = True,
):
    """
    Vẽ đồ thị có hướng (directed graph) từ ma trận p-value của Granger causality.

    Quy ước ma trận:
    - Hàng (index): biến gây tác động (cause)
    - Cột (columns): biến bị tác động (effect)
    - Nếu p_value_matrix.loc[cause, effect] < alpha => có cạnh cause -> effect

    Parameters
    ----------
    p_value_matrix : pd.DataFrame
        Ma trận p-value vuông, cùng bộ tên biến ở index và columns.
    alpha : float, default=0.05
        Ngưỡng ý nghĩa thống kê.
    title : str, default="Granger Causality Directed Graph"
        Tiêu đề biểu đồ.
    save_path : str | None, default=None
        Đường dẫn lưu ảnh. Nếu None thì chỉ hiển thị.
    show_edge_labels : bool, default=True
        Hiển thị nhãn p-value trên cạnh.
    """
    if p_value_matrix is None or p_value_matrix.empty:
        print("[plot_granger_causality_directed_graph] Ma trận p-value rỗng.")
        return

    if not isinstance(p_value_matrix, pd.DataFrame):
        raise TypeError("p_value_matrix phải là pandas.DataFrame")

    # Kiểm tra tính vuông và khớp nhãn index/columns
    if p_value_matrix.shape[0] != p_value_matrix.shape[1]:
        raise ValueError("p_value_matrix phải là ma trận vuông (N x N).")

    if list(p_value_matrix.index) != list(p_value_matrix.columns):
        raise ValueError("index và columns của p_value_matrix phải cùng thứ tự tên biến.")

    nodes = list(p_value_matrix.index)
    n_nodes = len(nodes)

    if n_nodes < 2:
        print("[plot_granger_causality_directed_graph] Cần ít nhất 2 biến để vẽ đồ thị.")
        return

    # --- Layout vòng tròn để dễ quan sát quan hệ hai chiều ---
    radius = 3.2
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    positions = {
        node: np.array([radius * np.cos(angle), radius * np.sin(angle)])
        for node, angle in zip(nodes, angles)
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    # Kích thước node dùng để căn chỉnh điểm bắt đầu/kết thúc mũi tên
    node_radius = 0.28
    edge_count = 0

    # --- Vẽ cạnh có hướng theo ngưỡng alpha ---
    for i, cause in enumerate(nodes):
        for j, effect in enumerate(nodes):
            if i == j:
                continue

            p_value = p_value_matrix.loc[cause, effect]
            if pd.isna(p_value) or p_value >= alpha:
                continue

            start = positions[cause]
            end = positions[effect]

            direction = end - start
            distance = np.linalg.norm(direction)
            if distance == 0:
                continue

            unit_vec = direction / distance

            # Dời điểm bắt đầu/kết thúc vào sát viền node để mũi tên đẹp hơn
            start_adj = start + unit_vec * node_radius
            end_adj = end - unit_vec * node_radius

            # Độ cong khác dấu theo cặp để giảm chồng chéo khi có cạnh ngược chiều
            curve_rad = 0.18 if i < j else -0.18

            # p-value càng nhỏ => cạnh càng đậm và dày
            strength = float(np.clip((alpha - p_value) / alpha, 0.0, 1.0))
            line_width = 1.0 + 2.8 * strength
            edge_color = plt.cm.Reds(0.35 + 0.65 * strength)

            arrow = FancyArrowPatch(
                posA=tuple(start_adj),
                posB=tuple(end_adj),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=line_width,
                color=edge_color,
                alpha=0.92,
                connectionstyle=f"arc3,rad={curve_rad}",
            )
            ax.add_patch(arrow)
            edge_count += 1

            if show_edge_labels:
                midpoint = (start_adj + end_adj) / 2.0
                normal = np.array([-unit_vec[1], unit_vec[0]])
                offset = normal * (0.18 if curve_rad > 0 else -0.18)
                text_pos = midpoint + offset

                ax.text(
                    text_pos[0],
                    text_pos[1],
                    f"p={p_value:.3f}",
                    fontsize=8,
                    color="#8B0000",
                    ha="center",
                    va="center",
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 0.4},
                )

    # --- Vẽ node trên cùng ---
    for node in nodes:
        x, y = positions[node]
        circle = plt.Circle((x, y), node_radius, facecolor="#4C78A8", edgecolor="white", linewidth=2.0, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, str(node), color="white", fontsize=10, fontweight="bold", ha="center", va="center", zorder=4)

    if edge_count == 0:
        ax.text(
            0,
            0,
            f"Không có cạnh nhân quả với alpha={alpha}",
            ha="center",
            va="center",
            fontsize=11,
            color="gray",
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-radius - 1.0, radius + 1.0)
    ax.set_ylim(-radius - 1.0, radius + 1.0)
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Visualizer] Granger directed graph saved at: {save_path}")

    plt.show()