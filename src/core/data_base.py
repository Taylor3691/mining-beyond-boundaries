from __future__ import annotations
from abc import ABC, abstractmethod
import glob
import os
import shutil
from tqdm import tqdm
from typing import TYPE_CHECKING

# Use a forward reference for type hinting the Visitor class
if TYPE_CHECKING:
    from src.image.analysis_deduplicate import AnalysisVisitor
    from src.image.preprocessing_remove_duplication import PreprocessingVisitor

class Object(ABC):
    @abstractmethod
    def load(path: str):
        pass

    @abstractmethod
    def save(path: str):
        pass

    @abstractmethod
    def info():
        pass

    @abstractmethod
    def clone():
        pass
    
    @abstractmethod
    def accept():
        pass

class TableDataset(Object):
    def load(path: str):
        return
    
    def save(path: str):
        return
    
class ImageDataset(Object):
    """
    Manages a dataset of images, represented by a list of file paths.
    This class can be "visited" by different processing objects.
    """
    def __init__(self):
        self.image_paths: list[str] = []
        self.name: str = "Untitled Image Dataset"

    def load(self, path: str) -> ImageDataset:
        """Loads all images from a directory and its subdirectories."""
        print(f"Loading image data from: {path}")
        self.name = os.path.basename(path)
        self.image_paths = []
        supported_formats = ('*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp')
        for fmt in supported_formats:
            self.image_paths.extend(glob.glob(os.path.join(path, '**', fmt), recursive=True))
        
        if not self.image_paths:
            print("No image files were found.")
        self.info()
        return self

    def save(self, path: str) -> None:
        """Saves the current dataset (remaining images) to a new directory."""
        print(f"Saving processed dataset to: {path}")
        os.makedirs(path, exist_ok=True)
        
        for img_path in tqdm(self.image_paths, desc=f"Copying images to '{path}'"):
            try:
                shutil.copy(img_path, os.path.join(path, os.path.basename(img_path)))
            except Exception as e:
                print(f"Could not copy file {img_path}: {e}")

    def info(self) -> None:
        """Prints summary information about the dataset."""
        print(f"--- Dataset Info: {self.name} ---")
        print(f"Total number of images: {len(self.image_paths)}")
        print("---------------------------------")

    def clone(self) -> ImageDataset:
        """Creates a deep copy of the dataset object."""
        new_dataset = ImageDataset()
        new_dataset.name = self.name + "_clone"
        new_dataset.image_paths = self.image_paths.copy()
        return new_dataset
    
    def accept(self, visitor: AnalysisVisitor | PreprocessingVisitor) -> None:
        """
        The entry point for the Visitor Pattern.
        It calls the appropriate visit method on the visitor, passing itself.
        """
        visitor.visitImageDataset(self)