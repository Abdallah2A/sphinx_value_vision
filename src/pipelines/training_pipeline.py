from zenml import pipeline
from src.steps.collect_data_step import run_spider
from src.steps.clean_data_step import clean_dataset
from src.steps.visualize_dataset_step import visualize_housing_data
from src.steps.train_model_step import train_model


@pipeline()
def training_pipeline():
    run_spider()

    clean_dataset()
    visualize_housing_data()

    train_model()
