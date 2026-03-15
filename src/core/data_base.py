from abc import ABC, abstractmethod
from utils import file
from config import DEFAULT_SIZE

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



class ImageDataset(Object):

    # Constructor
    def __init__(self, folder_path: str):
        self._folder_path = folder_path
        self._images = None
        self._class_idx = None
        self._labels = None
        self._paths = None
        self._size = 0
        self._shape = (0,0,0,0)
        self._image_size = DEFAULT_SIZE
        self._file_names = None
        return
    
    # Getter
    @property
    def images(self):
        return self._images, self._labels
    
    @property
    def class_idx(self):
        return self._class_idx

    @property
    def folder_path(self):
        return self._folder_path

    @property
    def image_paths(self):
        return self._paths
    
    @property
    def image_size(self):
        return self._image_size
    
    @property
    def dataset_shape(self):
        return self._shape

    # Setter
    @images.setter
    def images(self, value):
        if not value or len(value):
            raise ValueError("Images Folder cannot be empty")
        self._images = value
        self._size = len(self._images)
        self._shape = (self._size, *value[0].shape)


    @class_idx.setter
    def class_idx(self, value):
        if not value:
            raise ValueError("Class Index cannot be empty")
        self._class_idx = value

    @image_paths.setter
    def image_paths(self, value):
        if not value:
            raise ValueError("Image Paths Index cannot be empty")
        self._paths = value

    @folder_path.setter
    def folder_path(self, value):
        if not value:
            raise ValueError("Folder Path cannot be empty")
        self._folder_path = value

    # Method
    def load(self):
        X, Y, class_idx, paths, file_names = file.load_images(path=self._folder_path, image_size= self._image_size)
        self._images = X
        self._labels = Y
        self._class_idx = class_idx
        self._paths = paths
        self._size = len(X)
        self._shape = (len(X), *X[0].shape)
        self._file_names = file_names
        return
    
    def save(self, folder_path: str | None = None):
        folder_path = folder_path or self._folder_path
        file.save_images(folder_path, self._images)
        return
    
    def info(self):
        print("Metadata of Dataset")
        print("\tFolder Path:", self._folder_path if self._folder_path is not None else "Empty")
        print("\tTotal Images:", len(self._images) if self._images is not None else 0)
        print(f"\tImage Size:{self._image_size}")
        print(f"\tDataset Shape:{self._shape} (N,H,W,C)")

        print("\tName of file of 5 first sample")
        for i in range(1, min(self._size,5)):
            print(f"\t \t {i}: {self._file_names[i]}")
        
        return

class TableDataset(Object):
    def load(path: str):
        return
    
    def save(path: str):
        return