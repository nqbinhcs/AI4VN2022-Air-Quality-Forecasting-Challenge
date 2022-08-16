from abc import abstractmethod

class BaseModel():
    @abstractmethod
    def fit(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def save_model(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def load_model(self, *inputs):
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, *inputs):
        raise NotImplementedError