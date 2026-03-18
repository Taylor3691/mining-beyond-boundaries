import matplotlib.pyplot as plt
import seaborn as sns

# Task 8: Định nghĩa 2 hàm Visualize phân phối Histogram và KDE 

def plot_histogram(pixel_data, title_suffix=""):
    """
    Vẽ biểu đồ Histogram. 
    Sử dụng Subplots (1 hàng, 3 cột) để tách biệt 3 kênh R, G, B cho dễ nhìn.
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
    Vvẽ biểu đồ KDE.
    Sử dụng Subplots (1 hàng, 3 cột) để tách biệt 3 đường cong phân phối.
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