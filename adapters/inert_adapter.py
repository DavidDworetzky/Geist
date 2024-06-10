from abc import ABC, abstractmethod

class InertAdapter(ABC):
    '''
    Enumerable adapter that is not reflected as a callable adapter.
    '''
    @abstractmethod
    def enumerate_actions(self):
        pass
