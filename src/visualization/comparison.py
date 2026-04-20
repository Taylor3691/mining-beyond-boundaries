import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import numpy as np

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
    Vẽ biểu đồ so sánh trực quan giữa ảnh gốc và ảnh sau khi chuẩn hóa, 
    kèm theo phân phối Histogram và KDE cho từng kênh màu (RGB).
    
    Args:
        img_orig (np.ndarray): Mảng ảnh gốc (uint8).
        img_norm (np.ndarray): Mảng ảnh đã chuẩn hóa (float32).
        method (str): Tên phương pháp chuẩn hóa dùng trong tiêu đề.
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