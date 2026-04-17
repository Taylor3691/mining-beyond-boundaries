import config
import pandas as pd
from pathlib import Path
import cv2
import numpy as np
import pandas as pd 

def batch_loader(paths: list[str], batch_size: int = config.BATCH_SIZE):
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

"""
def load_table(path: str):
    try:
        df = pd.read_csv(path, low_memory=False)
        return df
    except Exception as e:
        raise IOError(f"Error loading CSV file from{path}: {e}")

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

def save_table(path: str, data: pd.DataFrame):
    try:
        data.to_csv(path, index=False) 
    except Exception as e:
        raise IOError(f"Error saving CSV file to {path}: {e}")
