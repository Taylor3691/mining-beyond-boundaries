from abc import ABC, abstractmethod
from src.core.data_base import ImageDataset

# Define a base Visitor for preprocessing tasks
class PreprocessingVisitor(ABC):
    @abstractmethod
    def visitImageDataset(self, dataset: ImageDataset) -> None:
        pass

class ImageDeduplicationRemoval(PreprocessingVisitor):
    """
    A Visitor that modifies an ImageDataset by removing duplicate images
    based on a pre-computed list of duplicate groups.
    """
    def __init__(self, duplicate_groups: list[list[str]]):
        """
        Initializes the removal process with the list of duplicates.

        Args:
            duplicate_groups (list[list[str]]): The output from an analysis visitor
                                                like ImageDeduplicationAnalysis.
        """
        self.duplicate_groups = duplicate_groups
        # Attributes for logging
        self._dataset_name = ""
        self._initial_count = 0
        self._final_count = 0
        self._status = "Not started"

    def visitImageDataset(self, dataset: ImageDataset) -> None:
        """
        Executes the removal process on the given dataset.
        """
        self._dataset_name = dataset.name
        self._initial_count = len(dataset.image_paths)
        self.log(message="Starting duplicate image removal process...")

        if not self.duplicate_groups:
            print("No duplicate groups provided. No changes will be made.")
            self._status = "Success (No action needed)"
            self._final_count = self._initial_count
            self.log()
            return

        # Identify all images to remove (keep the first one in each group)
        paths_to_remove = set()
        for group in self.duplicate_groups:
            paths_to_remove.update(group[1:])
        
        # Modify the dataset's image_paths list directly
        dataset.image_paths = [
            path for path in dataset.image_paths if path not in paths_to_remove
        ]
        
        self._final_count = len(dataset.image_paths)
        self._status = "Success"
        self.log()

    def log(self, message: str = ""):
        """Prints processing information to the console."""
        if message:
            print(f"\n--- LOG: {message} ---")
            return
            
        print("\n--- LOG: Removal Complete ---")
        print(f"1. Processing Step: Duplicate Image Removal")
        print(f"2. Target Dataset: {self._dataset_name}")
        print(f"3. Status: {self._status}")
        print("4. Removal Results:")
        print(f"   - Initial image count: {self._initial_count}")
        print(f"   - Images removed: {self._initial_count - self._final_count}")
        print(f"   - Final image count: {self._final_count}")
        print("---------------------------\n")