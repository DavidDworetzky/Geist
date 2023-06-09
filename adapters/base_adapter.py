from abc import ABC, abstractmethod

class BaseAdapter(ABC):
    @abstractmethod
    def enumerate_actions(self):
        pass
