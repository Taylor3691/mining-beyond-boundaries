from abc import ABC, abstractmethod
import cv2

class Object(ABC):
    @abstractmethod
    def load(path: str):
        pass

    @abstractmethod
    def save(path: str):
        pass



class ImageDataset(Object):
    def load(path: str):
        return
    
    def save(path: str):
        return


class TableDataset(Object):
    def load(path: str):
        return
    
    def save(path: str):
        return