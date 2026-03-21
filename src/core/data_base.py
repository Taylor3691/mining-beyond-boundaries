from __future__ import annotations
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