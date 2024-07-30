import logging
import pandas as pd
from zenml import step
import mlflow

from src.evaluation import MSE, RMSE, R2
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
        model: RegressorMixin,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Tuple[
        Annotated[float, 'r2'],
        Annotated[float, 'rmse'],
    ]:
    """
    Evaluating the model.

    Args:
        model (RegressorMixin): The trained model.
        X_test (pd.DataFrame): The testing data.
        y_test (pd.Series): The testing labels.
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("MSE", mse)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("R2", r2)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("RMSE", rmse)
        
        return r2, rmse

    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e