from image import ImageDataset
from core import Preprocessing

class ColorTransform(Preprocessing):
    def __init_subclass__(cls):
        return super().__init_subclass__()