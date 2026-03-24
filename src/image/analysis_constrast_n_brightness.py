import numpy as np
import pandas as pd
from core import Visualization
from image import ImageDataset
from visualization import relationship
from IPython.display import display, Markdown

class ContrastAndBrightness(Visualization):

    def __init__(self):
        self._dataset : ImageDataset | None = None
        self._df      : pd.DataFrame | None = None
        self._stats   : pd.DataFrame | None = None  # mean/var/median/IQR per class
        self._status  : str = "Not Run"

    # ------------------------------------------------------------------ #
    # Pipeline interface                                                   #
    # ------------------------------------------------------------------ #
    def run(self):
        if self._dataset is None:
            self._status = "Failed"
            return
        self.visitImageDataset(self._dataset)
        relationship.visualize_brightness_contrast_boxplot(self._df)
        self._analyze()

    def log(self):
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

    # ------------------------------------------------------------------ #
    # Service interface                                                    #
    # ------------------------------------------------------------------ #
    def visitImageDataset(self, obj: ImageDataset):
        """
        Task 15 — Tính brightness & contrast từng ảnh theo lớp.

        Công thức:
            gray       = 0.299·R + 0.587·G + 0.114·B   (ITU-R BT.601)
            brightness = mean(gray)   ∈ [0, 255]
            contrast   = std(gray)    ∈ [0, 127.5]
        """
        try:
            images, labels = obj.images
            idx_to_class   = {v: k for k, v in obj.class_idx.items()}

            records = []
            for img, label in zip(images, labels):
                gray = (
                    0.299 * img[:, :, 0].astype(np.float32)
                  + 0.587 * img[:, :, 1].astype(np.float32)
                  + 0.114 * img[:, :, 2].astype(np.float32)
                )
                records.append({
                    "class_name": idx_to_class.get(int(label), str(label)),
                    "brightness": float(np.mean(gray)),
                    "contrast"  : float(np.std(gray)),
                })

            self._df = pd.DataFrame(records, columns=["class_name", "brightness", "contrast"])

            # Tính stats per class: mean, var, median, IQR, n_outliers
            self._stats = self._compute_stats(self._df)

            self._dataset = obj
            self._status  = "Success"

        except Exception as e:
            self._status = f"Failed — {e}"

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #
    def _compute_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tính mean, var, median, IQR, n_outliers cho brightness & contrast theo class."""
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
        """Generate nhận xét tự động và render Markdown trong notebook."""
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
