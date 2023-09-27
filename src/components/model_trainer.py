import os
import sys
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from dataclasses import dataclass

## logging and Exception handling
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model



@dataclass
class ModelTrianerConfig:
    trained_model_file_path:str = os.path.join('artifacts','model.pkl')

class ModelTriner:
    def __init__(self):
        self.model_trainer_config = ModelTrianerConfig()
    
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Sependent and Independent variables from train and test data')
            X_train, y_train,X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'ElasticNet':ElasticNet()
            }

            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            print('Model Report')
            print('\n'+'*'*100 )
            logging.info(f'Model Report: {model_report}')

            # To get the best model score from model_report dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            print(f'Best Model Found, Model Name: {best_model_name} , R2 Score: {best_model_score}')
            print('\n'+'='*100 )
            logging.info(f'Best Model Found, Model Name: {best_model_name} , R2 Score: {best_model_score}')
            #logging.info('Hyperparameter tuning started for catboost')
            save_object(self.model_trainer_config.trained_model_file_path,
                    obj = best_model
                    )

        except Exception as e:
            logging.error('Error occured in Model Training')
            raise CustomException(e,sys)