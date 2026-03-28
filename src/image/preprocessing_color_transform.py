from image import ImageDataset
from core import Preprocessing
from config import SUPPORT_COLOR_SPACE

class ColorTransform(Preprocessing):
    def __init__(self, method: str = None):
        if method is None:
            raise ValueError("Cannot let the param blanked")
        elif method not in SUPPORT_COLOR_SPACE:
            raise ValueError("This Color Space is not supported")
        return
    
    def log(self):
        return super().log()
    
    def run(self, obj: ImageDataset):
        return super().run()
    
    def visitImageDataset():
        return
    

        