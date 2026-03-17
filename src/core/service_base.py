from abc import ABC, abstractmethod


class Pipeline(ABC):
    @abstractmethod
    def run(self):
        pass
    
    @abstractmethod
    def log(self):
        pass
    
class SubPipeline(Pipeline):
    def __init__(self):
        self._list = []
        return
    
    def addService(self, sub: Pipeline):
        self._list.append(sub)
        return

    def run():
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
    def __init__(self):
        return

class Testing(Service):
    def __init__(self):
        return