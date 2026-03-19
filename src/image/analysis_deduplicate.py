import sys
import os

# Đảm bảo Python nhận diện được thư mục gốc để import visualization và core
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
    
    The similarity between images is measured using the Hamming distance
    between their hash values. Images with a distance less than or equal
    to a predefined threshold are considered duplicates or near-duplicates.
    """

    def __init__(self, hash_size=8, similarity_threshold=10):
        """
        Initializes the analyzer.

        Args:
            hash_size (int): The size of the low-frequency DCT block used to generate the hash.
                             For example, hash_size=8 produces a 64-bit hash (8x8 block).
            
            similarity_threshold (int): The Hamming distance threshold used to determine
                                        whether two images are considered duplicates.
                                        
                                        Typical values:
                                        - 0–5   : Almost identical images
                                        - 5–10  : Very similar images
                                        - 10–15 : Moderately similar images
                                        - >15   : Likely different images
        """
        self._hash_size = hash_size
        self._threshold = similarity_threshold
        self._indices_to_remove = set()
        self._initial_count = 0
        self._final_count = 0
        self._status = "Not Run"
        self._dataset_path = "N/A"

    def _calculate_phash(self, image: np.ndarray) -> str:
        """
        Calculates the pHash (Perceptual Hash) for a single image.
        Steps:
        1. Resize image to 32x32
        2. Convert to grayscale
        3. Apply DCT
        4. Take top-left 8x8 DCT coefficients
        5. Compute mean and generate hash
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
        mean_val = np.mean(dct_low_freq[1:])  # bỏ phần tử [0,0] cho ổn định hơn

        # Step 6: Generate hash
        hash_str = "".join([
            '1' if val > mean_val else '0'
            for row in dct_low_freq
            for val in row
        ])
        return hash_str

    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        """Calculates the Hamming distance between two hash strings."""
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def _find_duplicates(self, images_data: list):
        """
        Finds all duplicate and near-duplicate images in the dataset.
        """
        num_images = len(images_data)
        if num_images < 2:
            return

        image_hashes = [self._calculate_phash(img) for img in images_data]
        checked_indices = [False] * num_images
        
        for i in range(num_images):
            if checked_indices[i]:
                continue
            
            for j in range(i + 1, num_images):
                if checked_indices[j]:
                    continue

                distance = self._hamming_distance(image_hashes[i], image_hashes[j])
                
                if distance <= self._threshold:
                    self._indices_to_remove.add(j)
                    checked_indices[j] = True

    def visitImageDataset(self, obj: ImageDataset):
        """
        Executes the entire analysis pipeline on an ImageDataset object.
        This is where Task 12 and Task 13 are implemented.
        """
        try:
            self._initial_count = obj._size
            self._dataset_path = obj._folder_path
            
            images_data, _ = obj.images

            self._find_duplicates(images_data)
            
            self._final_count = self._initial_count - len(self._indices_to_remove)
            
            # Uses a bar chart to visualize the number of images before and after deduplication
            plot_deduplicate_comparison(self._initial_count, self._final_count)
            
            self._status = "Success"
        except Exception as e:
            self._status = f"Failed: {e}"
            print(f"An error occurred in ImageDeduplication: {e}")
        
        self.log()
        return self._indices_to_remove
    
    def log(self):
        """
        Prints a summary log of the analysis process and its results.
        """
        print("\n--- Image Deduplication Analysis Log ---")
        print(f"1. Processing Step: Image Deduplication Analysis")
        print(f"2. Dataset Path: {self._dataset_path}")
        print(f"3. Status: {self._status}")
        print("4. Result Information:")
        if self._status == "Success":
            print(f"\t- Similarity Threshold (Hamming distance): {self._threshold}")
            print(f"\t- Initial Image Count: {self._initial_count}")
            print(f"\t- Duplicates Found: {len(self._indices_to_remove)}")
            print(f"\t- Estimated Final Image Count: {self._final_count}")
        print("----------------------------------------\n")
        
    def run(self, obj: ImageDataset):
        return self.visitImageDataset(obj)