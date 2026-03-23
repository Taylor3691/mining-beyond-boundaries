from abc import ABC, abstractmethod
from core import Object

class Pipeline(ABC):
    @abstractmethod
    def run(self):
        pass
    
    @abstractmethod
    def log(self):
        pass
    
class SubPipeline(Pipeline):
    def __init__(self, dataset: Object):
        self._list = []
        self._children = []
        self._dataset = dataset
        return
    
    def addService(self, sub: Pipeline):
        self._list.append(sub)
        return

    def run(self):
        for service in self._list:
            self._dataset.accept(service)
        return

class Service(Pipeline):
    """
    @abstractmethod
    def visitTabbleDataset():
        pass
    """

    @abstractmethod
    def visitImageDataset():
        pass

class Visualization(Service):
    pass

class Preprocessing(Service):
    @abstractmethod
    def fit():
        pass
    
    @abstractmethod
    def transform():
        pass

    @abstractmethod
    def fit_transform():
        pass

class Testing(Service):
    def __init__(self):
        return