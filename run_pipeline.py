from pipelines.training_pipeline import training_pipeline
from zenml.client import Client

if __name__ == '__main__':
    # Run the training pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline(data_path='./data/olist_customers_dataset.csv')

# mlflow ui --backend-store-uri "file:C:\Users\Asif\AppData\Roaming\zenml\local_stores\9445c78b-0009-4fc3-acc0-eff8e888c9dd\mlruns"