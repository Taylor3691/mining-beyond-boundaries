import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    Vẽ đồ thị đường thể hiện mối liên hệ giữa kích thước và chỉ số SSIM 
    """
    if len(sizes) != len(ssim_scores) or len(sizes) == 0:
        print("[Visualizer Error] Sizes and SSIM scores lists must have the same length and cannot be empty.")
        return

    # Sắp xếp size để đường vẽ chạy từ trái sang phải
    sorted_pairs = sorted(zip(sizes, ssim_scores))
    sorted_sizes = [pair[0] for pair in sorted_pairs]
    sorted_ssim = [pair[1] for pair in sorted_pairs]

    plt.figure(figsize=(9, 6))
    
    plt.plot(sorted_sizes, sorted_ssim, marker='o', linestyle='-', color='royalblue', 
             linewidth=2.5, markersize=8, label='SSIM Trend')

    for x, y in zip(sorted_sizes, sorted_ssim):
        plt.text(x, y + (max(sorted_ssim) * 0.005), f"{y:.4f}", 
                 ha='center', va='bottom', fontsize=11, fontweight='bold', color='darkblue')

    plt.title("Relationship Between Image Resize Dimension and SSIM", fontsize=15, pad=15)
    plt.xlabel("Target Image Size (NxN Pixels)", fontsize=12)
    plt.ylabel("Average SSIM Score", fontsize=12)
    
    plt.xticks(sorted_sizes)
    
    y_min, y_max = min(sorted_ssim), max(sorted_ssim)
    padding = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
    plt.ylim(y_min - padding, y_max + padding * 2)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Lưu và display lên màn hình 
    plt.savefig(save_path, dpi=300)
    print(f"\n[Visualizer] SSIM relationship curve successfully saved at: {save_path}")
    plt.show()