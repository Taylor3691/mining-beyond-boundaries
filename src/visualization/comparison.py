import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def plot_deduplicate_comparison(initial_count, final_count):
    """
    Uses a bar chart to visualize the number of images before and after deduplication.
    """
    labels = ['Before Removal', 'After Removal']
    counts = [initial_count, final_count]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, counts, color=['#1f77b4', '#ff7f0e'])
    plt.ylabel('Number of Images')
    plt.title('Image Count Before and After Deduplication')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom')

    plt.show()

def plot_normalization_comparison(img_orig: np.ndarray, img_norm: np.ndarray, method_name: str):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Cắt ảnh 8x8 ở trung tâm để hiển thị heatmap (ảnh to quá nhìn không rõ -.-)
    h, w = img_orig.shape[:2]
    cy, cx = h // 2, w // 2
    patches = [img_orig[cy-4:cy+4, cx-4:cx+4, 0], img_norm[cy-4:cy+4, cx-4:cx+4, 0]]
    images = [img_orig[:,:,0], img_norm[:,:,0]]
    
    is_diverging = img_norm.min() < 0
    cmaps = ['gray', 'coolwarm' if is_diverging else 'viridis']
    titles = ["Original", f"Normalized ({method_name})"]

    for i in range(2):
        # 1. Ảnh hiển thị
        axes[i, 0].imshow(images[i], cmap=cmaps[i])
        axes[i, 0].set_title(f"{titles[i]} Image")
        axes[i, 0].axis('off')

        # 2. Histogram
        sns.histplot(images[i].flatten(), bins=50, kde=(i==1), 
                     color='gray' if i==0 else 'blue', ax=axes[i, 1])
        axes[i, 1].set_title(f"{titles[i]} Distribution")

        # 3. Heatmap ROI
        sns.heatmap(patches[i], annot=True, fmt=".0f" if i==0 else ".3f", 
                    cmap="Blues" if i==0 else "coolwarm", 
                    center=0 if (i==1 and is_diverging) else None,
                    ax=axes[i, 2], cbar=True)
        axes[i, 2].set_title(f"{titles[i]} ROI (Center 8x8)")

    plt.suptitle(f"Method: {method_name}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
