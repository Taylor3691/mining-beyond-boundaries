import os
from abc import ABC, abstractmethod
from itertools import combinations
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import imagehash

from src.core.data_base import ImageDataset

# Define a base Visitor for analysis tasks
class AnalysisVisitor(ABC):
    @abstractmethod
    def visitImageDataset(self, dataset: ImageDataset) -> None:
        pass

class ImageDeduplicationAnalysis(AnalysisVisitor):
    """
    A Visitor that analyzes an ImageDataset to find duplicate and near-duplicate images
    using the pHash algorithm. It does not modify the dataset.
    """
    def __init__(self, threshold: int = 5):
        """
        Initializes the analysis with a pHash distance threshold.

        Args:
            threshold (int): The maximum Hamming distance to consider two images as duplicates.
        """
        self.threshold = threshold
        self.duplicate_groups: list[list[str]] = []
        # Attributes for logging
        self._dataset_name = ""
        self._initial_count = 0
        self._status = "Not started"

    def visitImageDataset(self, dataset: ImageDataset) -> None:
        """
        Executes the entire analysis process on the given dataset.
        This method finds duplicates, logs the process, and visualizes the results.
        """
        self._dataset_name = dataset.name
        self._initial_count = len(dataset.image_paths)
        self.log(message="Starting duplicate image analysis...")

        hashes = self._calculate_hashes(dataset.image_paths)
        self.duplicate_groups = self._find_duplicate_groups(hashes)

        if not self.duplicate_groups:
            self._status = "Success (No duplicates found)"
            self.log()
            return

        self._status = "Success (Duplicates found)"
        self._show_sample_duplicates()
        self._visualize_potential_results()
        self.log()
    
    def _calculate_hashes(self, image_paths: list[str]) -> dict[str, imagehash.ImageHash]:
        """Calculates and returns pHashes for all images."""
        hashes = {}
        print("Step 1/2: Calculating pHashes for all images...")
        for path in tqdm(image_paths, desc="Calculating pHashes"):
            try:
                hashes[path] = imagehash.phash(Image.open(path))
            except Exception as e:
                print(f"Error processing file {path}: {e}")
        return hashes

    def _find_duplicate_groups(self, hashes: dict) -> list[list[str]]:
        """Compares hashes and returns groups of duplicate images."""
        print("Step 2/2: Comparing hashes and grouping duplicates...")
        path_list = list(hashes.keys())
        adj = {path: [] for path in path_list}
        
        for path1, path2 in tqdm(combinations(path_list, 2), desc="Comparing Hashes"):
            if hashes.get(path1) is not None and hashes.get(path2) is not None:
                if hashes[path1] - hashes[path2] <= self.threshold:
                    adj[path1].append(path2)
                    adj[path2].append(path1)

        # Find connected components (groups of duplicates)
        groups = []
        visited = set()
        for path in path_list:
            if path not in visited:
                current_group = []
                q = [path]
                visited.add(path)
                while q:
                    node = q.pop(0)
                    current_group.append(node)
                    for neighbor in adj[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append(neighbor)
                if len(current_group) > 1:
                    groups.append(sorted(current_group))
        return groups

    def _visualize_potential_results(self):
        """Visualizes the number of images before and after potential removal."""
        potential_removed_count = sum(len(group) - 1 for group in self.duplicate_groups)
        final_count = self._initial_count - potential_removed_count
        
        labels = ['Original Count', 'Potential Final Count']
        counts = [self._initial_count, final_count]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, counts, color=['#1f77b4', '#ff7f0e'])
        plt.ylabel('Number of Images')
        plt.title('Potential Impact of Deduplication')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center')
        plt.show()

    def _show_sample_duplicates(self, num_groups_to_show: int = 2):
        """Displays a few sample groups of found duplicates."""
        print(f"\nDisplaying {min(num_groups_to_show, len(self.duplicate_groups))} sample duplicate groups...")
        for i, group in enumerate(self.duplicate_groups[:num_groups_to_show]):
            fig, axes = plt.subplots(1, len(group), figsize=(len(group) * 4, 4))
            if len(group) == 1: axes = [axes]
            fig.suptitle(f'Duplicate Group {i + 1}')
            for ax, img_path in zip(axes, group):
                try:
                    ax.imshow(Image.open(img_path))
                    ax.set_title(os.path.basename(img_path), fontsize=8)
                    ax.axis('off')
                except Exception:
                    ax.set_title("Load Error", fontsize=8, color='red')
            plt.show()

    def log(self, message: str = ""):
        """Prints processing information to the console."""
        if message:
            print(f"\n--- LOG: {message} ---")
            return
        
        print("\n--- LOG: Analysis Complete ---")
        print(f"1. Processing Step: Duplicate Image Analysis")
        print(f"2. Target Dataset: {self._dataset_name}")
        print(f"3. Status: {self._status}")
        print("4. Analysis Results:")
        print(f"   - Initial image count: {self._initial_count}")
        print(f"   - Duplicate groups found: {len(self.duplicate_groups)}")
        print(f"   - Total duplicate images found (excluding one original per group): {sum(len(g) - 1 for g in self.duplicate_groups)}")
        print("----------------------------\n")