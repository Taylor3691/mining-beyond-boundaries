import config
from pathlib import Path
import cv2
import numpy as np

def load_images(path: str, image_size=None):
    root = Path(path)
    class_names = config.CLASS_NAMES
    class_to_idx = config.CLASS_INDEX

    X, Y, file_names, paths = [], [], [], []
    exts = config.IMAGE_EXTS
    
    printed_original_size = False

    for class_name in class_names:
        count = 0
        class_dir = root / class_name
        if not class_dir.exists():
            continue

        for item in class_dir.rglob("*"):
            if item.suffix.lower() not in exts:
                continue

            img = cv2.imread(str(item))
            if img is None:
                continue
            if not printed_original_size:
                height, width, channels = img.shape
                # print(f"Kích thước ảnh gốc (Mẫu đầu tiên): {width}x{height} pixels, {channels} kênh màu")
                printed_original_size = True

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
            file_name = class_name + "_" + str(count) + item.suffix.lower()

            X.append(img)
            Y.append(class_to_idx[class_name])
            file_names.append(str(file_name))
            paths.append(str(item))

            count += 1

    Y = np.array(Y, dtype=np.int64)
    
    return X, Y, class_to_idx, paths, file_names

def load_table(path: str):
    return

def save_images(path: str, images: np.ndarray, file_names: list[str], is_classwise: bool = True):
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

def save_table(path: str):
    return