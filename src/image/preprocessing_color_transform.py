from image import ImageDataset
from core import Preprocessing
from config import DEFAULT_N_COMPONENTS, SUPPORT_COLOR_SPACE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

class ColorTransform(Preprocessing):
    def __init__(self, method: str = None, n: int = DEFAULT_N_COMPONENTS):
        if method is None:
            raise ValueError("Cannot let the method empty")
        elif method not in SUPPORT_COLOR_SPACE:
            raise ValueError("This color space is not support")
        
        self._scaler = StandardScaler()
        self._pca = PCA(n_components=n)
        self._method = method
        return
    
    def fit(self, arr: ):
        self._pca.fit()
        return
    def transform(self):
        self._pca.transform()
        return
        
    def fit_transform(self):
        self._pca.fit_transform()
        return self._pca.fit_transform()

    def log(self):
        return super().log()
    
    def visitImageDataset():
        return
    
    def run(self):
        return super().run()
    
