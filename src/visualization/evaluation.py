import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_color_space_evaluation(results: list, save_path: str = "color_space_eval.png"):
    """
    Vẽ biểu đồ kép so sánh Phương sai giải thích và F1-Score giữa các không gian màu.
    """
    if not results:
        print("[Visualizer Error] Danh sách kết quả rỗng.")
        return

    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Đánh giá Hiệu quả Tiền xử lý Không gian màu", fontsize=16, fontweight='bold')

    # Biểu đồ 1: Phương sai giải thích
    sns.barplot(data=df, x="Color_Space", y="Variance_Ratio_Sum", ax=axes[0], palette="viridis", edgecolor='black')
    axes[0].set_title("Khả năng giữ thông tin (Tỷ lệ Phương sai giải thích - PCA k=50)")
    axes[0].set_ylabel("Explained Variance Ratio Sum")
    axes[0].set_ylim(0, 1.0)
    for p in axes[0].patches:
        axes[0].annotate(format(p.get_height(), '.4f'), 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')

    # Biểu đồ 2: F1-Score
    sns.barplot(data=df, x="Color_Space", y="F1_Score", ax=axes[1], palette="magma", edgecolor='black')
    axes[1].set_title("Độ chính xác Mô hình (F1-Score trung bình)")
    axes[1].set_ylabel("F1-Score")
    for p in axes[1].patches:
        axes[1].annotate(format(p.get_height(), '.4f'), 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
def plot_normalization_learning_curves(results_dict: dict, save_path: str = "norm_learning_curves.png"):
    """
    Vẽ đường cong hội tụ (Learning Curves) so sánh độ mượt và tốc độ hội tụ của 4 phương pháp Chuẩn hóa.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    
    # results_dict có dạng: {"minmax_01": [0.2, 0.4, 0.5...], "zscore_global": [0.3, 0.6...]}
    for method, curve in results_dict.items():
        plt.plot(range(1, len(curve)+1), curve, marker='o', linewidth=2, label=method)
    
    plt.title("Tốc độ Hội tụ và Sự ổn định của các phương pháp Chuẩn hóa", fontsize=14, fontweight='bold')
    plt.xlabel("Số Epochs", fontsize=12)
    plt.ylabel("Độ chính xác (Accuracy) trên Test set", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Phương pháp Chuẩn hóa")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()