# Task 8: Thực hiện định nghĩa 2 hàm để nhận các thông tin Visualize 2 loại phân phối Histogram và KDE 
# (Lưu ý nhớ có chú thích và label x,y nếu có)

# Task 11: Kiểm tra Imbalance
import matplotlib.pyplot as plt

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