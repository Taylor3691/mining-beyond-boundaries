import sys
import os
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from core import Visualization
from image.dataset import ImageDataset
from visualization import plot_deduplicate_comparison

class ImageDeduplication(Visualization):
    """
    Analyzes and identifies duplicate or near-duplicate images within an ImageDataset.
    It uses the pHash (Perceptual Hash) algorithm to generate compact representations
    of images based on their visual content in the frequency domain (via DCT).
    """

    def __init__(self, hash_size=8, similarity_threshold=10):
        """
        Khởi tạo lớp phân tích và tìm kiếm ảnh trùng lặp.

        Input:
            hash_size: Kích thước của mã băm (mặc định 8x8).
            similarity_threshold: Ngưỡng khoảng cách Hamming để coi là trùng lặp (mặc định 10).
        """
        self._hash_size = hash_size
        self._threshold = similarity_threshold
        self._indices_to_remove = []
        self._initial_count = 0
        self._final_count = 0
        self._status = "Not Run"
        self._dataset_path = "N/A"

    def _calculate_phash(self, image: np.ndarray) -> str:
        """
        Tính toán mã băm pHash (Perceptual Hash) cho một ảnh.

        Input:
            image: Ma trận ảnh đầu vào (Numpy array).

        Output:
            Chuỗi ký tự '0' và '1' đại diện cho mã băm.
        """
        # Step 1: Resize to 32x32
        resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        # Step 2: Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # Step 3: Convert to float32 and apply DCT
        gray = np.float32(gray)
        dct = cv2.dct(gray)
        # Step 4: Take top-left 8x8 (low frequency)
        dct_low_freq = dct[:8, :8]
        # Step 5: Compute mean (exclude DC coefficient [0,0] optionally)
        mean_val = np.mean(dct_low_freq[1:])  
        # Step 6: Generate hash
        hash_str = "".join(['1' if val > mean_val else '0' for row in dct_low_freq for val in row])
        return hash_str

    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        Tính khoảng cách Hamming giữa hai mã băm.

        Input:
            hash1, hash2: Hai chuỗi mã băm cần so sánh.

        Output:
            Số lượng vị trí khác biệt giữa hai mã băm.
        """
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def _find_duplicates(self, images_data: list):
        """
        Tìm kiếm các ảnh trùng lặp trong một danh sách ảnh cho trước (Xử lý in-memory).

        Input:
            images_data: Danh sách các mảng numpy chứa dữ liệu ảnh.
        """
        num_images = len(images_data)
        if num_images < 2: return
        image_hashes = [self._calculate_phash(img) for img in images_data]
        checked_indices = [False] * num_images
        indices_set = set()
        
        for i in range(num_images):
            if checked_indices[i]: continue
            for j in range(i + 1, num_images):
                if checked_indices[j]: continue
                distance = self._hamming_distance(image_hashes[i], image_hashes[j])
                if distance <= self._threshold:
                    indices_set.add(j)
                    checked_indices[j] = True
        self._indices_to_remove = sorted(list(indices_set))

    def run_batch(self, dataset: ImageDataset):
        """
        Quét và tìm ảnh trùng lặp theo batch để tiết kiệm bộ nhớ.

        Input:
            dataset: Đối tượng ImageDataset.

        Output:
            Danh sách index các ảnh bị coi là trùng lặp cần xóa.
        """
        valid_hashes = []
        total_paths = len(dataset.image_paths)
        self._initial_count = total_paths
        self._dataset_path = dataset.folder_path
        
        print(f"[PROCESS] Đang tính toán pHash cho {total_paths} ảnh...")

        for batch_images, batch_indices in dataset.load():
            for img, path_idx in zip(batch_images, batch_indices):
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                valid_hashes.append((path_idx, self._calculate_phash(img_bgr)))
            
            print(f" -> Đã băm: {len(valid_hashes)} / {total_paths} ảnh...", end='\r')
            del batch_images
            gc.collect()

        print("\n[PROCESS] Thu thập hoàn tất. Đang so khớp mã băm...")
        
        indices_to_remove = set()
        n = len(valid_hashes)

        for i in range(n):
            idx_i, hash_i = valid_hashes[i]
            if idx_i in indices_to_remove: continue

            for j in range(i + 1, n):
                idx_j, hash_j = valid_hashes[j]
                if idx_j in indices_to_remove: continue

                if self._hamming_distance(hash_i, hash_j) <= self._threshold:
                    indices_to_remove.add(idx_j)

        self._indices_to_remove = sorted(list(indices_to_remove))
        self._final_count = total_paths - len(self._indices_to_remove)
        self._status = "Success"
        
        self.log()
        return self._indices_to_remove

    def visitImageDataset(self, obj: ImageDataset):
        """
        Xử lý tìm ảnh trùng lặp cụ thể trên đối tượng ImageDataset.

        Input:
            obj: Đối tượng ImageDataset.

        Output:
            Danh sách các index trùng lặp.
        """
        try:
            self._initial_count = obj._size
            self._dataset_path = obj._folder_path
            images_data, _ = obj.images
            self._find_duplicates(images_data)
            self._final_count = self._initial_count - len(self._indices_to_remove)
            plot_deduplicate_comparison(self._initial_count, self._final_count)
            self._status = "Success"
        except Exception as e:
            self._status = f"Failed: {e}"
            print(f"An error occurred in ImageDeduplication: {e}")
        self.log()
        return self._indices_to_remove
    
    def log(self):
        """
        In log chi tiết về quá trình tìm kiếm ảnh trùng lặp.
        """
        print("\n--- Image Deduplication Analysis Log ---")
        print(f"1. Processing Step: Image Deduplication Analysis")
        print(f"2. Dataset Path: {self._dataset_path}")
        print(f"3. Status: {self._status}")
        print("4. Result Information:")
        if self._status == "Success":
            print(f"\t- Similarity Threshold: {self._threshold}")
            print(f"\t- Initial Image Count: {self._initial_count}")
            print(f"\t- Duplicates Found: {len(self._indices_to_remove)}")
            print(f"\t- Estimated Final Image Count: {self._final_count}")
        print("----------------------------------------\n")

    def run(self, obj: ImageDataset):
        """
        Thực thi quy trình phân tích trùng lặp.

        Input:
            obj: Đối tượng ImageDataset.
        """
        if isinstance(obj, ImageDataset):
            self.visitImageDataset(obj)
        return