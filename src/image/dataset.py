from core import Object, Service
from utils import file
from config import DEFAULT_SIZE, PATH_FOLDER_IMAGE_RAW, BATCH_SIZE, CLASS_INDEX, CLASS_NAMES

class ImageDataset(Object):

    def __init__(self, path: str | None = PATH_FOLDER_IMAGE_RAW):
        self._folder_path = path
        self._size = 0
        self._shape = (0,0,0,0)
        self._image_size = DEFAULT_SIZE

        self._images = []
        self._class_idx = CLASS_INDEX
        self._paths, self._labels, self._file_names = file.load_image_paths(path)
        return
    
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

    @images.setter
    def images(self, value):
        if not value or len(value) == 0:
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
    """ load() version 1
    def load(self):
        Y, class_idx, paths, file_names = file.load_images(path=self._folder_path, image_size= self._image_size)
        # self._images = X
        self._labels = Y
        self._class_idx = class_idx
        self._paths = paths
        self._size = len(Y)
        # self._shape = (len(X), *X[0].shape)
        self._file_names = file_names
        return
    """

    def load(self, class_name: str = None):
        if class_name is None:
            return file.batch_loader(paths=self._paths, batch_size=BATCH_SIZE)

        if class_name not in CLASS_NAMES:
            raise ValueError("Class Name is not defined")

        class_paths = [
            path for path, label in zip(self._paths, self._labels)
            if label == CLASS_INDEX[class_name]
        ]

        return file.batch_loader(paths=class_paths, batch_size=BATCH_SIZE)

    def save(self, folder_path: str | None = None, is_classwise: bool= True):
        folder_path = folder_path or self._folder_path
        file.save_images(folder_path, self._images, self._file_names, is_classwise)
        return
    
    def info(self):
        print("Metadata of Dataset")
        print("\tFolder Path:", self._folder_path if self._folder_path is not None else "Empty")
        print("\tTotal Images:", len(self._images) if self._images is not None else 0)
        print(f"\tImage Size:{self._image_size}")
        print(f"\tDataset Shape:{self._shape} (N,H,W,C)")

        print("\tName of file of 5 first sample")
        for i in range(0, min(self._size,5)):
            print(f"\t \t {i}: {self._file_names[i]}")
        
        return
    
    def clone(self):
        dataset_clone = ImageDataset(self._folder_path)
        dataset_clone._folder_path = self._folder_path
        dataset_clone._size = self._size
        dataset_clone._shape = self._shape
        dataset_clone._image_size = self._image_size

        dataset_clone._class_idx = self._class_idx.copy()
        dataset_clone._labels = self._labels.copy()
        dataset_clone._file_names = self._file_names.copy()
        dataset_clone._paths = self._paths.copy()

        dataset_clone._images = [img.copy() for img in self._images]

        return dataset_clone
    
    def accept(self, service: Service):
        service.run(self)
        return
