import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# Task 11: Kiểm tra Imbalance
# ==========================================
def plot_class_distribution(class_names: list, counts: list):
    """
    Hàm vẽ Bar Chart so sánh số lượng ảnh giữa các lớp:
    - Nhấn mạnh lớp nhiều nhất (Đỏ) và ít nhất (Xanh lá)
    - Vẽ đường cảnh báo nếu có lớp có số lượng vượt quá 3 lần lớp nhỏ nhất
    """
    if not class_names or not counts:
        print("No data available to plot")
        return

    # Tìm giá trị lớn nhất, nhỏ nhất
    max_count = max(counts)
    min_count = min(counts)
    
    max_idx = counts.index(max_count)
    min_idx = counts.index(min_count)

    # Khởi tạo màu sắc (mặc định màu xanh dương nhạt cho tất cả các lớp)
    colors = ['skyblue'] * len(class_names)
    
    # Nhấn mạnh 2 lớp có số lượng lớn nhất và bé nhất
    colors[max_idx] = 'salmon'     # Lớn nhất
    colors[min_idx] = 'lightgreen' # Bé nhất

    # Tiến hành vẽ barlot 
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, counts, color=colors, edgecolor='black')

    # Hiển thị số lượng mẫu cụ thể trên đầu mỗi cột
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + (max_count * 0.02), 
                 int(yval), ha='center', va='bottom', fontweight='bold')

    # Kiểm tra có lớp nào vượt mức 3x so với lớp ít nhất không 
    title = "Image Class Distribution"
    if min_count > 0 and max_count > 3 * min_count:
        threshold_3x = 3 * min_count
        # Vẽ một đường gạch ngang đứt nét tại vị trí 3x
        plt.axhline(y=threshold_3x, color='red', linestyle='--', linewidth=2, 
                    label=f'3x Minority Threshold ({threshold_3x} samples)')
        plt.legend()
        title += "\n(Warning: Severe Data Imbalance)"

    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Classes", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.xticks(rotation=45, ha='right') 
    
    plt.tight_layout()
    
    # Lưu file biểu đồ 
    output_file = "class_distribution_chart.png"
    plt.savefig(output_file, dpi=300)
    print(f"\n[Visualizer] Chart successfully saved at: {output_file}")
    
    plt.show()
    plt.close()

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
    plt.close()

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
    plt.subplots_adjust(bottom=0.25) # Chừa khoảng trống phía dưới cho legend
    plt.show()
    plt.close()