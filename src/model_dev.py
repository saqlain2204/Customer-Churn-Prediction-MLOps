import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for all models
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model.
        Args:
            X_train (pd.DataFrame): The training data.
            y_train (pd.Series): The training labels.
        
        Returns:
            None
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression Models
    """
    def train(self, X_train, y_train):
        """
        Trains the model.
        Args:
            X_train (pd.DataFrame): The training data.
            y_train (pd.Series): The training labels.
        Returns:
            None
        """
        try:
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            logging.info("Model trained.")
            return reg
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e
