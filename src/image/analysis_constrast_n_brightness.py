import numpy as np
import pandas as pd
from core import Visualization
from image import ImageDataset
from visualization import relationship
from IPython.display import display, Markdown

class ContrastAndBrightness(Visualization):

    def __init__(self):
        """
        Khởi tạo lớp phân tích độ sáng và độ tương phản của tập dữ liệu ảnh.
        
        Input:
            Không có.
        
        Output:
            None.
        """
        self._dataset : ImageDataset | None = None
        self._df      : pd.DataFrame | None = None
        self._stats   : pd.DataFrame | None = None  # mean/var/median/IQR per class
        self._status  : str = "Not Run"

    def run(self):
        """
        Thực thi quy trình phân tích và hiển thị kết quả.
        
        Input:
            Không có.
        
        Output:
            None.
        """
        if self._dataset is None:
            self._status = "Failed"
            return
        self.visitImageDataset(self._dataset)
        relationship.visualize_brightness_contrast_boxplot(self._df)
        self._analyze()

    def log(self):
        """
        In log kết quả phân tích ra màn hình.
        
        Input:
            Không có.
        
        Output:
            None.
        """
        print("=" * 55)
        print("Step    : Analysis Contrast & Brightness")
        print(f"Dataset : {self._dataset._folder_path if self._dataset else 'None'}")
        print(f"Status  : {self._status}")
        if self._df is not None:
            print("\n── Global (toàn bộ data) ──")
            for metric in ["brightness", "contrast"]:
                vals = self._df[metric]
                print(f"  {metric:12s}  mean={vals.mean():.2f}  var={vals.var():.2f}")

            print("\n── Theo từng class ──")
            print(self._stats.to_string())
        print("=" * 55)

    def visitImageDataset(self, obj: ImageDataset):
        """
        Duyệt qua ImageDataset, tính toán độ sáng và độ tương phản cho từng ảnh theo batch.
        
        Input:
            obj: Đối tượng dữ liệu đầu vào cần được xử lý.
        
        Output:
            None.
        """
        try:
            # Map index sang tên class để hiển thị
            idx_to_class = {v: k for k, v in obj.class_idx.items()}
            records = []

            for batch_images, batch_indices in obj.load():
                
                for img, idx in zip(batch_images, batch_indices):
                    # Lấy nhãn thực tế dựa trên index trả về từ loader
                    label_val = obj._labels[idx]
                    class_name = idx_to_class.get(int(label_val), str(label_val))

                    # Chuyển sang grayscale bằng Vectorization
                    gray = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
                    
                    records.append({
                        "class_name": class_name,
                        "brightness": float(np.mean(gray)),
                        "contrast"  : float(np.std(gray)),
                    })

            # Kiểm tra nếu không có dữ liệu
            if not records:
                self._status = "Failed — No images loaded"
                return

            # Tạo DataFrame từ list kết quả
            self._df = pd.DataFrame(records, columns=["class_name", "brightness", "contrast"])

            # Tính toán thống kê (giữ nguyên hàm _compute_stats của bạn)
            self._stats = self._compute_stats(self._df)
            self._dataset = obj
            self._status  = "Success"

        except Exception as e:
            self._status = f"Failed — {str(e)}"

    def _compute_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính toán các chỉ số thống kê (mean, var, median, IQR, outliers) cho độ sáng và độ tương phản.
        
        Input:
            df: Dữ liệu đầu vào dùng cho bước phân tích hoặc biến đổi.
        
        Output:
            Giá trị trả về của hàm.
        """
        rows = []
        for class_name, grp in df.groupby("class_name"):
            row = {"class_name": class_name}
            for metric in ["brightness", "contrast"]:
                vals = grp[metric]
                q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                iqr    = q3 - q1
                n_out  = int(((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)).sum())
                row[f"{metric}_mean"]      = round(vals.mean(), 2)
                row[f"{metric}_var"]       = round(vals.var(),  2)
                row[f"{metric}_median"]    = round(vals.median(), 2)
                row[f"{metric}_IQR"]       = round(iqr, 2)
                row[f"{metric}_n_outliers"]= n_out
            rows.append(row)
        return pd.DataFrame(rows).set_index("class_name")

    def _analyze(self):
        """
        Hiển thị nhận xét kết quả dưới dạng Markdown.
        
        Input:
            Không có.
        
        Output:
            None.
        """
        s = self._stats

        # ── Brightness ──────────────────────────────────────────────────
        bright_median = s["brightness_median"]
        brightest     = bright_median.idxmax()
        darkest       = bright_median.idxmin()

        bright_var    = s["brightness_var"]
        most_var_b    = bright_var.idxmax()
        least_var_b   = bright_var.idxmin()

        bright_iqr    = s["brightness_IQR"]
        large_iqr_b   = bright_iqr[bright_iqr > bright_iqr.median()].index.tolist()
        small_iqr_b   = bright_iqr[bright_iqr <= bright_iqr.median()].index.tolist()

        bright_out    = s["brightness_n_outliers"]
        noisy_b       = bright_out[bright_out > bright_out.median()].index.tolist()

        # ── Contrast ────────────────────────────────────────────────────
        contrast_var  = s["contrast_var"]
        most_var_c    = contrast_var.idxmax()
        least_var_c   = contrast_var.idxmin()

        contrast_out  = s["contrast_n_outliers"]
        noisy_c       = contrast_out[contrast_out > contrast_out.median()].index.tolist()

        # ── Render Markdown ─────────────────────────────────────────────
        md = f"""
## Nhận xét: Phân tích Độ sáng & Độ tương phản theo Lớp

### 1. Độ sáng (Brightness — Mean Intensity)

**Median theo lớp:**
- Class **{brightest}** có median cao nhất `({bright_median[brightest]:.2f})` → ảnh sáng nhất trung bình.
- Class **{darkest}** có median thấp nhất `({bright_median[darkest]:.2f})` → ảnh tối nhất trung bình.

**Phương sai (Variance) theo lớp:**
- Class **{most_var_b}** có variance lớn nhất `({bright_var[most_var_b]:.2f})` → độ sáng phân tán rộng, ảnh trong lớp không đồng đều.
- Class **{least_var_b}** có variance nhỏ nhất `({bright_var[least_var_b]:.2f})` → ảnh trong lớp khá đồng đều về độ sáng.

**IQR (boxplot) theo lớp:**
- Các class có IQR lớn: **{', '.join(large_iqr_b)}** → hộp boxplot rộng, phân bố độ sáng biến động đáng kể trong lớp.
- Các class có IQR nhỏ: **{', '.join(small_iqr_b)}** → hộp boxplot hẹp, ảnh trong lớp có độ sáng tập trung.

**Outliers:**
- Các class có nhiều outlier về độ sáng: **{', '.join(noisy_b) if noisy_b else 'Không có'}** → tồn tại ảnh có độ sáng bất thường, có thể gây nhiễu khi huấn luyện.

---

### 2. Độ tương phản (Contrast — Std Intensity)

**Phương sai (Variance) theo lớp:**
- Class **{most_var_c}** có variance lớn nhất `({contrast_var[most_var_c]:.2f})` → độ tương phản giữa các ảnh trong lớp dao động nhiều.
- Class **{least_var_c}** có variance nhỏ nhất `({contrast_var[least_var_c]:.2f})` → độ tương phản khá đồng nhất trong lớp.

**Outliers:**
- Các class có nhiều outlier về độ tương phản: **{', '.join(noisy_c) if noisy_c else 'Không có'}** → một số ảnh có độ tương phản rất khác biệt so với phần còn lại của lớp.
"""
        display(Markdown(md))
