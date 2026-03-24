import os
import shutil
from core import Preprocessing
from image.dataset import ImageDataset


class RemoveDuplication(Preprocessing):
    """
    A service to remove duplicate images from an ImageDataset based on a
    pre-determined set of indices. It can also optionally move/copy those 
    duplicate files to a separate directory.
    """
    def __init__(self, indices_to_remove: set, output_dir: str = None):
        """
        Initializes the service with a set of indices to remove.
        
        Args:
            indices_to_remove (set): A set containing the integer indices of images to be removed.
            output_dir (str, optional): The directory path where duplicate images should be copied.
                                        If None, files are only removed from memory, not copied.
        """
        if not isinstance(indices_to_remove, set):
            raise TypeError("indices_to_remove must be a 'set' for efficient lookup.")
        self._indices_to_remove = indices_to_remove
        self._output_dir = output_dir
        self._status = "Not Run"
        self._removed_count = 0
        self._copied_count = 0

    # =========================================================================
    # IMPLEMENT ABSTRACT METHODS TỪ CLASS CHA (BẮT BUỘC ĐỂ TRÁNH TYPEERROR)
    # =========================================================================
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def visitImageDataset(self, obj: ImageDataset):
        """Implement chuẩn theo pattern Visitor của hệ thống"""
        self.run(obj)
        return self
    # =========================================================================

    def run(self, obj: ImageDataset):
        """
        Executes the removal of duplicate images from the ImageDataset object.
        If output_dir is provided, copies the files to that directory first.
        This method modifies the dataset object in-place.
        """
        initial_count = obj._size
        if initial_count == 0:
            print("Dataset is empty. Nothing to remove.")
            self._status = "Success"
            self.log(0, 0)
            return

        try:
            # 1. Xử lý copy file vật lý (Nếu có truyền output_dir)
            if self._output_dir and len(self._indices_to_remove) > 0:
                os.makedirs(self._output_dir, exist_ok=True)
                for idx in self._indices_to_remove:
                    source_path = obj._paths[idx]
                    file_name = obj._file_names[idx]
                    destination_path = os.path.join(self._output_dir, file_name)
                    
                    try:
                        shutil.copy(source_path, destination_path)
                        self._copied_count += 1
                    except Exception as copy_err:
                        print(f"Warning: Failed to copy {file_name}: {copy_err}")

            # 2. Xử lý loại bỏ ảnh khỏi Dataset trên RAM
            indices_to_keep = [i for i in range(initial_count) if i not in self._indices_to_remove]
            
            obj._images = [obj._images[i] for i in indices_to_keep]
            obj._labels = [obj._labels[i] for i in indices_to_keep]
            obj._paths = [obj._paths[i] for i in indices_to_keep]
            obj._file_names = [obj._file_names[i] for i in indices_to_keep]
            
            new_size = len(obj._images)
            obj._size = new_size
            obj._shape = (new_size, *obj._images[0].shape) if new_size > 0 else (0, 0, 0, 0)
            
            self._removed_count = initial_count - new_size
            self._status = "Success"
            
        except Exception as e:
            self._status = f"Failed: {e}"
            print(f"An error occurred during removal process: {e}")
        
        self.log(initial_count, obj._size)
            
    def log(self, initial_count, final_count):
        """Prints a summary log of the removal process."""
        print("\n--- Remove Duplication Preprocessing Log ---")
        print(f"1. Processing Step: Preprocessing - Remove Duplicates")
        print(f"2. Status: {self._status}")
        if self._status == "Success":
            print(f"\t- Initial Image Count: {initial_count}")
            if self._output_dir:
                print(f"\t- Files Copied to '{self._output_dir}': {self._copied_count}")
            print(f"\t- Number of Images Removed from RAM: {self._removed_count}")
            print(f"\t- Final Image Count: {final_count}")
        print("------------------------------------------\n")