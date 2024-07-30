import logging

import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy
from typing import Annotated
from typing import Tuple

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test']
]:
    '''
    Cleans the data and divides it into training and testing sets.

    Args:
        df (pd.DataFrame): The ingested data.
    
    Returns:
        X_train (pd.DataFrame): The training data.
        X_test (pd.DataFrame): The testing data.
        y_train (pd.Series): The training labels.
        y_test (pd.Series): The testing labels.

    
    '''
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning and division complete.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in cleaning data: {e}")
        raise e