import config
from pathlib import Path
import cv2
import numpy as np

def load_images(path: str, image_size = (128,128)):
    root = Path(path)
    class_names = config.CLASS_NAMES
    class_to_idx = config.CLASS_INDEX

    X,Y, file_names ,paths = [],[],[],[]

    exts = config.IMAGE_EXTS
    for class_name in class_names:
        count  = 0
        class_dir = root / class_name
        if not class_dir.exists():
            continue

        for item in class_dir.rglob("*"):
            if item.suffix.lower() not in exts :
                continue

            img = cv2.imread(str(item))
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
            file_name = class_name + "_" + str(count)

            X.append(img)
            Y.append(class_to_idx[class_name])
            file_names.append(str(file_name))
            paths.append(str(item))

            count+=1

    X = np.array(X, dtype=np.uint8)
    Y = np.array(Y, dtype=np.int64)
    return X, Y, paths, class_to_idx, file_names

def load_table(path: str):
    return

def save_images(path: str, images):
    return

def save_table(path: str):
    return

