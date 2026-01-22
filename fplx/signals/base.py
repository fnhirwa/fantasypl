from abc import ABC, abstractmethod


class BaseSignal(ABC):
    @abstractmethod
    def generate_signal(self, data): ...
