from abc import ABC, abstractmethod

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