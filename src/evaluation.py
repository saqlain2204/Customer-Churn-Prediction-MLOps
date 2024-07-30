import logging
from abc import ABC, abstractmethod
import numpy as np
from numpy.core.multiarray import array as array
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluating our models
    """
    @abstractmethod
    def calculate_scores(self, y_ture: np.array, y_pred: np.array):
        """
        Calculates the scores of the model.
        Args:
            y_true (np.array): The true labels.
            y_pred (np.array): The predicted labels.
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """
    Evaluation streategy that uses Mean Squared Error
    """
    def calculate_scores(self, y_ture: np.array, y_pred: np.array):
        try:
            logging.info("Calculating Mean Squared Error.")
            mse = mean_squared_error(y_ture, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error("Error in calculating Mean Squared Error.")
            raise e
    
class R2(Evaluation):
    """
    Evaluation streategy that uses R2 Score
    """
    def calculate_scores(self, y_ture: np.array, y_pred: np.array):
        try:
            logging.info("Calculating R2 Score.")
            r2 = r2_score(y_ture, y_pred)
            logging.info(f"R2 Score: {r2}")
            return r2
        except Exception as e:
            logging.error("Error in calculating R2 Score.")
            raise e

class RMSE(Evaluation):
    """
    Evaluation streategy that uses Root Mean Squared Error
    """
    def calculate_scores(self, y_ture: np.array, y_pred: np.array):
        try:
            logging.info("Calculating Root Mean Squared Error.")
            rmse = np.sqrt(mean_squared_error(y_ture, y_pred))
            logging.info(f"RMSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error("Error in calculating Root Mean Squared Error.")
            raise e
    