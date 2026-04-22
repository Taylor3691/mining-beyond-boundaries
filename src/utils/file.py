import config
import pandas as pd
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

def batch_loader(paths: list[str], batch_size: int = config.BATCH_SIZE):
    """
    Tải ảnh theo từng batch từ danh sách đường dẫn.

    Input:
        paths: Danh sách đường dẫn tới các file ảnh.
        batch_size: Số lượng ảnh tối đa mỗi batch (mặc định từ config).

    Output:
        Generator trả về tuple (batch_images, batch_indices) cho mỗi batch.
    """
    for i in range(0, len(paths), batch_size):
        batch = []
        index = list(range(i, min(i + batch_size, len(paths))))

        for j in range(i, min(i + batch_size, len(paths))):
            img = cv2.imread(paths[j], cv2.IMREAD_COLOR)

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch.append(img)
        
        index = list(range(i, min(i + batch_size, len(paths))))
        yield batch, index
    return

def load_image_paths(path: str):
    """
    Quét thư mục gốc và trả về danh sách đường dẫn ảnh, nhãn, và tên file theo từng class.

    Input:
        path: Đường dẫn thư mục gốc chứa các thư mục con theo tên lớp.

    Output:
        Tuple (paths, labels, filenames):
            paths: Danh sách đường dẫn tuyệt đối tới từng file ảnh.
            labels: Danh sách chỉ số lớp tương ứng.
            filenames: Danh sách tên file đã đặt lại theo quy tắc <class>_<index>.<ext>.
    """
    root = Path(path)
    class_names = config.CLASS_NAMES
    class_idx = config.CLASS_INDEX
    paths, labels, filenames = [], [], []

    exts = config.IMAGE_EXTS
    

    for class_name in class_names:
        class_dir = root / class_name
        count = 0
        if not class_dir.exists():
            continue

        for item in class_dir.rglob("*"):
            if item.suffix.lower() not in exts:
                continue
            paths.append(str(item))
            labels.append(class_idx[class_name])
            filenames.append(class_name + "_" + str(count) + item.suffix.lower())
            count+=1

    return paths, labels, filenames

def load_table(path: str):
    """
    Đọc file CSV vào DataFrame.

    Input:
        path: Đường dẫn tới file CSV.

    Output:
        pd.DataFrame chứa dữ liệu bảng đã nạp.
    """
    try:
        df = pd.read_csv(path, low_memory=False)
        return df
    except Exception as e:
        raise IOError(f"Error loading CSV file from{path}: {e}")

def save_images(path: str, images: np.ndarray, file_names: list[str], is_classwise: bool = True):
    """
    Lưu danh sách ảnh vào thư mục đích, có thể phân theo class.

    Input:
        path: Đường dẫn thư mục đích để lưu ảnh.
        images: Mảng numpy chứa các ảnh cần lưu.
        file_names: Danh sách tên file tương ứng cho mỗi ảnh.
        is_classwise: True nếu tạo thư mục con theo tên class (mặc định True).

    Output:
        None (lưu file ảnh vào ổ đĩa).
    """
    folder_save = Path(path)
    folder_save.mkdir(parents=True, exist_ok=True)
    if is_classwise:
        for image, name in zip(images, file_names):
            class_name = name.split("_")[0]
            save_path = folder_save / Path(class_name)
            save_path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path/name), image)
    else:
        for image, name in zip(images, file_names):
            cv2.imwrite(str(folder_save/ name), image)
    return

def save_table(path: str, data: pd.DataFrame, file_name: str = "processed_data.csv"):
    """
    Lưu DataFrame ra file CSV tại thư mục chỉ định.

    Input:
        path: Đường dẫn thư mục đích.
        data: DataFrame cần lưu.
        file_name: Tên file CSV đầu ra (mặc định 'processed_data.csv').

    Output:
        None (lưu file CSV vào ổ đĩa).
    """
    try:
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        data.to_csv(folder / file_name, index=False)
    except Exception as e:
        raise IOError(f"Error saving CSV file to {path}: {e}")
    
def jaccard_similarity(list1: list, list2: list) -> float:
    """Tính chỉ số Jaccard Similarity giữa 2 tập hợp index"""
    set1, set2 = set(list1), set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0

