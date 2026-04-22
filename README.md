# Data Mining Preprocessing Project

Đồ án tiền xử lý dữ liệu cho môn **Data Mining 2026**, thực hiện khảo sát và cài đặt các kỹ thuật tiền xử lý trên 3 nhóm dữ liệu: **Image**, **Table** và **Time Series**. Đồ án bao gồm phân tích dữ liệu thô, xử lý giá trị thiếu, phát hiện & xử lý outlier, chuẩn hóa, mã hóa đặc trưng, xử lý mất cân bằng lớp, giảm chiều dữ liệu và nhiều kỹ thuật tiền xử lý khác.

- Báo cáo thực nghiệm và mô tả chi tiết đồ án tại [PDF](./docs/report.pdf)
- Github: [mining-beyond-boundaries](https://github.com/Taylor3691/mining-beyond-boundaries)
- Dữ liệu: [Drive](https://drive.google.com/drive/folders/1CeU6lBn2ezqJF7EqhpnOa-deqcfZg7uC?usp=drive_link) 
- Phân công: [Google Sheet](https://docs.google.com/spreadsheets/d/1rtoR6K0IoPLpgj9U4qKEQ-JOqeFfI-DcvqiA9hehfnI/edit?usp=sharing)
---

## Mục lục

1. [Thông tin nhóm](#1-thông-tin-nhóm)
2. [Mô tả tập dữ liệu](#2-mô-tả-tập-dữ-liệu)
3. [Chức năng](#3-chức-năng)
   - [Phần 1 — Tiền xử lý ảnh](#phần-1--tiền-xử-lý-ảnh)
   - [Phần 2 — Tiền xử lý dữ liệu bảng](#phần-2--tiền-xử-lý-dữ-liệu-bảng)
   - [Phần 3 — Tiền xử lý chuỗi thời gian](#phần-3--tiền-xử-lý-chuỗi-thời-gian)
4. [Yêu cầu](#4-yêu-cầu)
5. [Clone và cài môi trường](#5-clone-và-cài-môi-trường)
   - [5.1. Clone repo](#51-clone-repo)
   - [5.2. Tạo virtual environment](#52-tạo-virtual-environment)
   - [5.3. Cài dependencies](#53-cài-dependencies)
6. [Tải dataset từ Kaggle](#6-tải-dataset-từ-kaggle)
   - [6.1. Download trực tiếp trên web](#61-cách-1-download-trực-tiếp-trên-web)
   - [6.2. Dùng Kaggle CLI](#62-cách-2-dùng-kaggle-cli)
7. [Cấu trúc dữ liệu bắt buộc](#7-cấu-trúc-dữ-liệu-bắt-buộc)
8. [Kiểm tra nhanh dữ liệu](#8-kiểm-tra-nhanh-dữ-liệu-đã-đặt-đúng-chưa)
9. [Cách chạy đồ án](#9-cách-chạy-đồ-án)
   - [9.1. Chạy bằng Notebook](#91-cách-khuyến-nghị-chạy-bằng-notebook)
   - [9.2. Chạy module Python trực tiếp](#92-chạy-một-số-module-python-trực-tiếp)
10. [Cách sử dụng](#10-cách-sử-dụng)
11. [Lỗi thường gặp và cách xử lý](#11-lỗi-thường-gặp-và-cách-xử-lý)

---

## 1. Thông tin nhóm

| STT | MSSV | Họ và tên |
|:---:|:--------:|-------------------|
| 1 | 23120146 | Hoàng Ngọc |
| 2 | 23122025 | Phạm Ngọc Duy |
| 3 | 23122032 | Nguyễn Việt Hùng |
| 4 | 23120266 | Võ Trần Duy Hoàng |
| 5 | 23122021 | Bùi Duy Bảo |

---

## 2. Mô tả tập dữ liệu

### Dữ liệu Image — Animals-10

| Thuộc tính | Chi tiết |
|---|---|
| **Nguồn** | [Kaggle — Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) |
| **Số lượng ảnh** | ~26.000 ảnh |
| **Số lớp** | 10 (`dog`, `cat`, `horse`, `spider`, `butterfly`, `chicken`, `sheep`, `cow`, `squirrel`, `elephant`) |
| **Định dạng** | JPEG / PNG, kích thước không đồng nhất |
| **Mô tả** | Tập ảnh động vật thu thập từ Google Images, được phân thành 10 thư mục theo loài. Tên thư mục gốc bằng tiếng Ý, cần đổi sang tiếng Anh trước khi sử dụng. |
| **Ứng dụng trong đồ án** | Phân tích mất cân bằng lớp, resize & chuẩn hóa ảnh, augmentation, giảm chiều (PCA, t-SNE), kiểm định Kolmogorov-Smirnov |

### Dữ liệu Table — Building Permit Applications

| Thuộc tính | Chi tiết |
|---|---|
| **Nguồn** | [Kaggle — Building Permit Applications Data](https://www.kaggle.com/datasets/aparnashastry/building-permit-applications-data) |
| **Số bản ghi** | ~198.000 dòng |
| **Số cột** | 43 cột |
| **Định dạng** | CSV (`Building_Permits.csv`) |
| **Mô tả** | Dữ liệu đơn xin cấp phép xây dựng tại San Francisco từ 01/01/2013 trở đi, bao gồm thông tin về loại công trình, địa chỉ, chi phí ước tính, ngày nộp/cấp phép, trạng thái, v.v. Dữ liệu chứa nhiều giá trị thiếu và outlier, phù hợp cho bài toán tiền xử lý. |
| **Ứng dụng trong đồ án** | Phân tích & xử lý missing values, phát hiện & xử lý outlier, mã hóa đặc trưng (Label, One-Hot, Target, Frequency Encoding), chuẩn hóa (Min-Max, Z-score), xử lý mất cân bằng (SMOTE, ADASYN, Under-sampling), giảm chiều (PCA, t-SNE) |

### Dữ liệu Time Series — COVID-19 Time Series

| Thuộc tính | Chi tiết |
|---|---|
| **Nguồn** | [Kaggle — COVID-19 Time Series Data](https://www.kaggle.com/datasets/niketchauhan/covid-19-time-series-data) |
| **Phạm vi** | Dữ liệu chuỗi thời gian COVID-19 toàn cầu (confirmed, deaths, recovered) |
| **Định dạng** | CSV (`time-series-19-covid-combined.csv`) |
| **Mô tả** | Dữ liệu chuỗi thời gian tổng hợp về số ca xác nhận, tử vong và hồi phục do COVID-19 theo quốc gia/vùng lãnh thổ và theo ngày. |
| **Ứng dụng trong đồ án** | Kiểm định tính dừng (ADF, KPSS), phân tích xu hướng & mùa vụ (seasonal decomposition), xử lý missing values chuỗi thời gian, làm mượt (smoothing), chuẩn hóa chuỗi thời gian |

---

## 3. Chức năng

Đồ án thực hiện 3 phần tương ứng với 3 loại dữ liệu (Phần 1 — Ảnh, Phần 2 — Bảng, Phần 3 — Chuỗi thời gian). Bảng dưới đây tóm tắt các tác vụ đã thực hiện cho từng phần. Ngoài ra còn nhiều kỹ thuật khác được trình bày chi tiết trong báo cáo.

### Phần 1 — Tiền xử lý ảnh 

| Tác vụ | Nội dung chi tiết |
|---|---|
| **Phân tích thống kê tập dữ liệu** | Perceptual Hashing (pHash), phân tích mất cân bằng lớp (class imbalance), kiểm định Kolmogorov-Smirnov (KS test) |
| **Cài đặt kỹ thuật tiền xử lý & đo lường** | Resize, chuẩn hóa ảnh, augmentation; đánh giá chất lượng bằng SSIM và PSNR |
| **Giảm chiều & đánh giá** | Ablation study, phân tích không gian đặc trưng bằng PCA và t-SNE |

### Phần 2 — Tiền xử lý dữ liệu bảng 

| Tác vụ | Nội dung chi tiết |
|---|---|
| **EDA thống kê** | Kiểm định phân phối, kiểm định MCAR test cho dữ liệu thiếu |
| **Phát hiện ngoại lai & mã hóa đặc trưng** | Phát hiện outlier (IQR, Z-score), mã hóa đặc trưng đầu vào (Label, One-Hot, Target, Frequency Encoding) |
| **Lựa chọn đặc trưng & đánh giá** | Feature selection, đánh giá mô hình bằng Cross-Validation F1-score |

### Phần 3 — Tiền xử lý chuỗi thời gian

| Tác vụ | Nội dung chi tiết |
|---|---|
| **Kiểm định tính dừng & phân rã** | Kiểm định ADF, KPSS; phân rã STL (Seasonal-Trend decomposition using LOESS) |
| **Phát hiện dị thường & xây dựng đặc trưng** | Phát hiện anomaly trong chuỗi thời gian, xây dựng ma trận đặc trưng (feature matrix) |
| **Dự báo & kiểm định nhân quả** | Dự báo bằng Random Forest Regressor, kiểm định nhân quả Granger (Granger causality) |

---

## 4. Yêu cầu

- Python 3.10+ (khuyến nghị 3.11)
- Git
- VS Code + Python extension + Jupyter extension (nếu chạy notebook)

---

## 5. Clone và cài môi trường

### 5.1. Clone repo

```bash
git clone https://github.com/Taylor3691/mining-beyond-boundaries.git
cd mining-beyond-boundaries
```

### 5.2. Tạo virtual environment

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 5.3. Cài dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```
---

## 6. Tải dataset từ Kaggle

Project dùng 3 nguồn dataset:

1. Image: https://www.kaggle.com/datasets/alessiocorrado99/animals10
2. Table: https://www.kaggle.com/datasets/aparnashastry/building-permit-applications-data
3. Time series: https://www.kaggle.com/datasets/niketchauhan/covid-19-time-series-data

### 6.1. Cách 1: Download trực tiếp trên web

1. Mở từng link Kaggle bên trên.
2. Bấm `Download`.
3. Giải nén và đặt file/folder vào đúng vị trí (xem mục 4 bên dưới).

### 6.2. Cách 2: dùng Kaggle CLI

#### Bước A: Tạo API token

1. Vào Kaggle -> Account -> Create New API Token.
2. File `kaggle.json` sẽ được tải về.
3. Đặt file vào:
   - Windows: `%USERPROFILE%\\.kaggle\\kaggle.json`
   - macOS/Linux: `~/.kaggle/kaggle.json`

#### Bước B: Cài kaggle cli

```bash
pip install kaggle
```

#### Bước C: Tải 3 dataset

```bash
kaggle datasets download -d alessiocorrado99/animals10 -p data --unzip
kaggle datasets download -d aparnashastry/building-permit-applications-data -p data/table --unzip
kaggle datasets download -d niketchauhan/covid-19-time-series-data -p data/time_series --unzip
```

Sau khi unzip xong, bạn cần sắp xếp lại thư mục theo đúng cấu trúc ở mục 4.

---

## 7. Cấu trúc dữ liệu bắt buộc

Sau khi tải dữ liệu, đảm bảo thư mục `data/` có dạng:

```text
data/
  image/
    butterfly/
    cat/
    chicken/
    cow/
    dog/
    elephant/
    horse/
    sheep/
    spider/
    squirrel/
  table/
    Building_Permits.csv
  time_series/
    time-series-19-covid-combined.csv
  small/
    ... (dữ liệu mẫu nhỏ để test nhanh)
```

Quan trọng:
- Project đăng ký class image theo tên tiếng Anh: `dog, spider, cat, sheep, elephant, chicken, butterfly, cow, squirrel, horse`.
- Dataset `animals10` có thể xuất hiện folder tên tiếng Ý (ví dụ `cane`, `gatto`, ...). Nếu gặp trường hợp này, bạn cần đổi tên về tiếng Anh như trên.

Mapping gợi ý (Ý -> Anh):
- `cane -> dog`
- `gatto -> cat`
- `cavallo -> horse`
- `pecora -> sheep`
- `elefante -> elephant`
- `gallina -> chicken`
- `farfalla -> butterfly`
- `mucca -> cow`
- `scoiattolo -> squirrel`
- `ragno -> spider`

---

## 8. Kiểm tra nhanh dữ liệu đã đặt đúng chưa

Windows PowerShell:

```powershell
Test-Path .\data\table\Building_Permits.csv
Test-Path .\data\time_series\time-series-19-covid-combined.csv
Get-ChildItem .\data\image -Directory | Select-Object Name
```

Nếu kết quả là `True` cho 2 file CSV và có đủ 10 class ảnh, bạn đã sẵn sàng chạy.

---

## 9. Cách chạy đồ án

## 9.1. Cách khuyến nghị: chạy bằng Notebook

1. Mở folder project bằng VS Code.
2. Chọn Python interpreter là `.venv`.
3. Mở notebook trong `notebooks/`.
4. Run cell từ trên xuống dưới.

Notebook mẫu để test nhanh:
- `notebooks/Hoang-image.imbalance.analysis.ipynb`
- `notebooks/Duy-table.analysis.missing.value.ipynb`
- `notebooks/Bao-time.series.analysis.stationarity.testing.ipynb`

Lưu ý:
- Nhiều notebook đã có đoạn `sys.path.append('../src')`, vì vậy nếu mở notebook đúng vị trí trong repo là chạy được.

### 9.2. Chạy một số module Python trực tiếp

Từ thư mục gốc project, bạn có thể chạy:

```bash
python src/image/analysis_imbalance_class.py
python src/image/preprocessing_resize.py
python src/table/preprocessing_detect_outlier.py
```

---

## 10. Cách sử dụng

- Nhóm image:
  - Tạo `ImageDataset`
  - Gọi các service `analysis_*` hoặc `preprocessing_*`
- Nhóm table:
  - Tạo `TableDataset` từ `Building_Permits.csv`
  - Chạy preprocessing/analysis từ `src/table/`
- Nhóm time series:
  - Tạo `TimeSeriesDataset` từ `time-series-19-covid-combined.csv`
  - Chạy preprocessing/analysis từ `src/time_series/`

Kết quả thường là:
- log trên terminal/notebook
- bảng thống kê
- biểu đồ phân tích

---

## 11. Lỗi thường gặp và cách xử lý

### Lỗi 1: `ModuleNotFoundError: No module named ...`

Fix:
1. Kiểm tra đã active `.venv` chưa.
2. Chạy lại:

```bash
pip install -r requirements.txt
```

3. Nếu chạy notebook, đảm bảo đã chọn đúng kernel `.venv`.

4. Nếu vẫn lỗi module, thêm path `src` bằng `os.path` + `sys.path`:

Notebook (thêm ở cell đầu):

```python
import os
import sys

src_path = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))
if src_path not in sys.path:
    sys.path.append(src_path)
```

Script Python (thêm ở đầu file):

```python
import os
import sys

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.append(src_path)
```

### Lỗi 2: Không tìm thấy file dataset

Fix:
1. Kiểm tra đúng tên file:
   - `Building_Permits.csv`
   - `time-series-19-covid-combined.csv`
2. Kiểm tra đúng folder:
   - `data/table/`
   - `data/time_series/`
   - `data/image/<10_classes>/`

### Lỗi 3: Chạy chậm/tốn RAM khi xử lý image

Fix:
1. Test trước với `data/small/`.
2. Sau khi ổn định mới chạy full `data/image/`.

---
