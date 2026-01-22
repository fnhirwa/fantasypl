from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y=None): ...

    @abstractmethod
    def predict(self, X): ...
