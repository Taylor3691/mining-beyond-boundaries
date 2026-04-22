# Data Mining Preprocessing Project

Đồ án tiền xử lý dữ liệu cho môn Data Mining 2026, gồm 3 nhóm dữ liệu:
- Image
- Table
- Time series

README này hướng dẫn bạn chạy dự án từ đầu đến cuối:
1. Cài môi trường
2. Tải dataset từ Kaggle
3. Đặt dataset đúng thư mục
4. Chạy notebooks và scripts
5. Cách sử dụng cơ bản

---

## 1. Yêu cầu

- Python 3.10+ (khuyến nghị 3.11)
- Git
- VS Code + Python extension + Jupyter extension (nếu chạy notebook)

---

## 2. Clone và cài môi trường

### 2.1. Clone repo

```bash
git clone https://github.com/Taylor3691/data-mining-preprocessing-project.git
cd data-mining-preprocessing-project
```

### 2.2. Tạo virtual environment

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

### 2.3. Cài dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```
---

## 3. Tải dataset từ Kaggle

Project dùng 3 nguồn dataset:

1. Image: https://www.kaggle.com/datasets/alessiocorrado99/animals10
2. Table: https://www.kaggle.com/datasets/aparnashastry/building-permit-applications-data
3. Time series: https://www.kaggle.com/datasets/niketchauhan/covid-19-time-series-data

### 3.1. Cách 1: Download trực tiếp trên web

1. Mở từng link Kaggle bên trên.
2. Bấm `Download`.
3. Giải nén và đặt file/folder vào đúng vị trí (xem mục 4 bên dưới).

### 3.2. Cách 2: dùng Kaggle CLI

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

## 4. Cấu trúc dữ liệu bắt buộc

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

## 5. Kiểm tra nhanh dữ liệu đã đặt đúng chưa

Windows PowerShell:

```powershell
Test-Path .\data\table\Building_Permits.csv
Test-Path .\data\time_series\time-series-19-covid-combined.csv
Get-ChildItem .\data\image -Directory | Select-Object Name
```

Nếu kết quả là `True` cho 2 file CSV và có đủ 10 class ảnh, bạn đã sẵn sàng chạy.

---

## 6. Cách chạy đồ án

## 6.1. Cách khuyến nghị: chạy bằng Notebook

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

### 6.2. Chạy một số module Python trực tiếp

Từ thư mục gốc project, bạn có thể chạy:

```bash
python src/image/analysis_imbalance_class.py
python src/image/preprocessing_resize.py
python src/table/preprocessing_detect_outlier.py
```

---

## 7. Cách sử dụng

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

## 8. Lỗi thường gặp và cách xử lý

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
