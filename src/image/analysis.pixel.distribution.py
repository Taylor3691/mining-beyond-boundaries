from abc import ABC, abstractmethod
import numpy as np

class PixelDistribution(ABC):
    @abstractmethod
    def fit():
        pass

    @abstractmethod
    def count():
        pass

class HistogramDistribution(PixelDistribution):
    def fit():
        return
    
    def count():
        return
    

class KDEDistribution(PixelDistribution):
    def fit():
        return
    
    def count():
        return

