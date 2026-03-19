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