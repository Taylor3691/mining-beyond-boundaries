from image import ImageDataset
from core import Preprocessing
import numpy as np
import cv2
import random
import os
import concurrent.futures
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

def horizontal_flip(image: np.ndarray) -> np.ndarray:
    """Lật ngang ảnh theo chiều dọc."""
    return cv2.flip(image, 1)

def rotate_image(image: np.ndarray, angle: float = None) -> np.ndarray:
    """Xoay ảnh một góc bất kỳ hoặc xác định."""
    if angle is None:
        angle = random.uniform(-180, 180)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def random_crop(image: np.ndarray, crop_factor: float = 0.8) -> np.ndarray:
    """Cắt và resize ảnh ngẫu nhiên theo tỷ lệ."""
    h, w = image.shape[:2]
    ch, cw = int(h * crop_factor), int(w * crop_factor)
    if h == ch or w == cw:
        return image
    y = random.randint(0, h - ch)
    x = random.randint(0, w - cw)
    cropped = image[y:y+ch, x:x+cw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def add_gaussian_noise(image: np.ndarray, mean: float = 0, std: float = None) -> np.ndarray:
    """Thêm nhiễu Gaussian vào ảnh."""
    if std is None:
        std = random.uniform(10.0, 30.0)
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def adjust_brightness_contrast(image: np.ndarray,
                                alpha: float = None,
                                beta: float = None) -> np.ndarray:
    """Điều chỉnh độ sáng và tương phản của ảnh."""
    if alpha is None:
        alpha = random.uniform(0.8, 1.2)
    if beta is None:
        beta = random.uniform(-30, 30)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

class DataAugmentation(Preprocessing):
    def __init__(self,
                 target_count: int = None,
                 apply_original: bool = True):
        """
        Khởi tạo lớp tăng cường dữ liệu và cân bằng lớp.

        Input:
            target_count: Số lượng mẫu mục tiêu cho mỗi lớp. Nếu None, lấy theo lớp có nhiều mẫu nhất.
            apply_original: Có giữ lại ảnh gốc hay không (mặc định True).
        """
        self._target_count = target_count
        self._apply_original = apply_original

        self._aug_funcs = [
            horizontal_flip,
            rotate_image,
            random_crop,
            add_gaussian_noise,
            adjust_brightness_contrast,
        ]

        self._method = "Data Augmentation Pipeline (Class-Balanced)"
        self._stats: Dict[str, Any] = {}
        self._max_workers = min(16, (os.cpu_count() or 1) + 4)

    @property
    def stats(self):
        return self._stats

    def _apply_random_aug(self, image: np.ndarray) -> np.ndarray:
        """Áp dụng ngẫu nhiên một số phép biến đổi lên ảnh."""
        num_ops = random.randint(1, 3)
        chosen = random.sample(self._aug_funcs, min(num_ops, len(self._aug_funcs)))
        aug = image.copy()
        for fn in chosen:
            aug = fn(aug)
        return aug

    def _augment_record(self,
                        record: Tuple,
                        aug_index: int) -> Tuple:
        """
        Tạo một mẫu record mới đã qua biến đổi từ record gốc.

        Input:
            record: Tuple chứa (img, fname, label, path).
            aug_index: Chỉ số của ảnh biến đổi để đặt tên file.

        Output:
            Tuple record mới (aug_img, new_fname, label, new_path).
        """
        img, fname, label, path = record
        name, ext = os.path.splitext(fname)
        aug_img = self._apply_random_aug(img)
        new_fname = f"{name}_aug_{aug_index}{ext}"
        new_path  = path.replace(fname, new_fname)
        return (aug_img, new_fname, label, new_path)

    def fit(self, arr: list):
        """Cập nhật số lượng mẫu đã xử lý."""
        if not arr:
            raise ValueError("Input batch is empty. Cannot fit data.")
        self._stats["processed_count"] = (
            self._stats.get("processed_count", 0) + len(arr)
        )

    def transform(self, arr: list) -> list:
        """Thực hiện cân bằng và tăng cường dữ liệu."""
        return self._balance_records(arr)

    def fit_transform(self, arr: list) -> list:
        """Hàm fit và transform kết hợp."""
        self.fit(arr)
        results = self.transform(arr)
        generated = len(results) - (len(arr) if self._apply_original else 0)
        self._stats["generated_count"] = (
            self._stats.get("generated_count", 0) + generated
        )
        return results

    def _balance_records(self, records: List[Tuple]) -> List[Tuple]:
        """Cân bằng số mẫu giữa các lớp bằng augmentation."""
        by_class: Dict[Any, List[Tuple]] = defaultdict(list)
        for rec in records:
            by_class[rec[2]].append(rec)

        class_counts = {lbl: len(recs) for lbl, recs in by_class.items()}
        target = self._target_count or max(class_counts.values())

        self._stats["class_counts_before"] = dict(class_counts)
        self._stats["target_count"] = target

        balanced: List[Tuple] = []
        total_generated = 0

        for label, recs in by_class.items():
            current_count = len(recs)
            needed = max(0, target - current_count)

            if self._apply_original:
                balanced.extend(recs)

            aug_counter = 0
            source_pool = recs
            pool_idx = 0
            for _ in range(needed):
                src_rec = source_pool[pool_idx % len(source_pool)]
                pool_idx += 1
                balanced.append(self._augment_record(src_rec, aug_counter))
                aug_counter += 1
                total_generated += 1

        self._stats["generated_count"] = (
            self._stats.get("generated_count", 0) + total_generated
        )

        after_counts: Dict[Any, int] = Counter(rec[2] for rec in balanced)
        self._stats["class_counts_after"] = dict(after_counts)

        return balanced

    def run(self, obj: ImageDataset):
        """Thực thi quy trình tăng cường dữ liệu."""
        if isinstance(obj, ImageDataset):
            self.visitImageDataset(obj)

    def visitImageDataset(self, obj: ImageDataset):
        """Xử lý cụ thể cho ImageDataset."""
        try:
            all_records: List[Tuple] = []
            for batch_imgs, batch_indices in obj.load():
                if not batch_imgs:
                    continue
                for i, idx in enumerate(batch_indices):
                    all_records.append((
                        batch_imgs[i],
                        obj._file_names[idx],
                        obj._labels[idx],
                        obj._paths[idx],
                    ))

            self._stats["processed_count"] = len(all_records)
            balanced = self._balance_records(all_records)

            obj.images       = [r[0] for r in balanced]
            obj._file_names  = [r[1] for r in balanced]
            obj._labels      = [r[2] for r in balanced]
            obj.image_paths  = [r[3] for r in balanced]

            self._stats["status"] = "Success"
        except Exception as e:
            self._stats["status"] = f"Failed ({e})"
        finally:
            self.log()

    def log(self):
        """In thông tin thống kê về quá trình xử lý."""
        print(f"Method: {self._method}")
        for key, value in self._stats.items():
            print(f"  - {key}: {value}")