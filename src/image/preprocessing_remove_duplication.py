from core import Service
from image.dataset import ImageDataset


class RemoveDuplication(Service):
    """
    A service to remove duplicate images from an ImageDataset based on a
    pre-determined set of indices.
    """
    def __init__(self, indices_to_remove: set):
        """
        Initializes the service with a set of indices to remove.
        Args:
            indices_to_remove (set): A set containing the integer indices of images to be removed.
        """
        if not isinstance(indices_to_remove, set):
            raise TypeError("indices_to_remove must be a 'set' for efficient lookup.")
        self._indices_to_remove = indices_to_remove
        self._status = "Not Run"
        self._removed_count = 0

    def run(self, obj: ImageDataset):
        """
        Executes the removal of duplicate images from the ImageDataset object.
        This method modifies the dataset object in-place.
        """
        initial_count = obj._size
        if initial_count == 0:
            print("Dataset is empty. Nothing to remove.")
            self._status = "Success"
            self.log(0, 0)
            return

        try:
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
            print(f"\t- Number of Images Removed: {self._removed_count}")
            print(f"\t- Final Image Count: {final_count}")
        print("------------------------------------------\n")