from abc import ABC, abstractmethod
import cv2
from utils import file

class Object(ABC):
    @abstractmethod
    def load(path: str):
        pass

    @abstractmethod
    def save(path: str):
        pass



class ImageDataset(Object):

    # Constructor
    def __init__(self, folder_path: str):
        self._folder_path = folder_path
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

    # Setter
    @images.setter
    def images(self, value):
        if not value:
            raise ValueError("Iamges cannot be empty")
        self._images = value

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
        X, Y, class_idx, paths = file.load_images(path=self.path)
        self._images = X
        self._labels = Y
        self._class_idx = class_idx
        self._paths = paths
        return
    
    def save(self, folder_path: str | None = None):
        folder_path = folder_path or self._folder_path
        file.save_images(folder_path, self._images)
        return


class TableDataset(Object):
    def load(path: str):
        return
    
    def save(path: str):
        return