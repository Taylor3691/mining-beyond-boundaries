import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Task 11: Kiểm tra Imbalance
import matplotlib.pyplot as plt

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

# Task 8

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
    plt.close()

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
        sns.kdeplot(pixel_data[:, i], color=colors[i], fill=True, alpha=0.3, ax=axes[i], linewidth=2)
        axes[i].set_title(f"{labels[i]} Channel", fontsize=13, color=colors[i])
        axes[i].set_xlabel('Giá trị Pixel (0 - 255)', fontsize=12)
        axes[i].grid(axis='y', alpha=0.3, linestyle='--')
        
    axes[0].set_ylabel('Mật độ phân phối', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    plt.close()
