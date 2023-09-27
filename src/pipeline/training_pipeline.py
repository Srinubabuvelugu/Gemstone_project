import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTriner








## run the Data Ingestion

if __name__ == '__main__':
    obj=DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr,_v    = data_transformation.initiate_data_Transforamtion(train_data_path,test_data_path)
    model_trainer = ModelTriner()
    model_trainer.initiate_model_training(train_arr,test_arr)