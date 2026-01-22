from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    @abstractmethod
    def solve(self, players): ...
