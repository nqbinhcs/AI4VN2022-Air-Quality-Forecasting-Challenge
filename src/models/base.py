from abc import abstractmethod

class BaseModel():
    @abstractmethod
    def fit(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    @abstractmethod
    def save_model(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError
