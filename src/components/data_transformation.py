import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer  ## Handling Missing Values
from sklearn.preprocessing import StandardScaler, OrdinalEncoder ## Handling Feature Scalling and  Ordinal Encoder
from sklearn.compose import ColumnTransformer
## pipe lines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
## logging and Exception handling
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@ dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    train_path = os.path.join('artifacts','train.csv')
    test_path = os.path.join('artifacts','test.csv')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation_object(self):
        logging.info('Data Transformation initiated')
        try:

            ## Definr the which Column should be Ordinal-Encoder(i.e Categorical columns) and which column should be Scaled(i.e Numerical columns) 
            categorical_columns = ['cut', 'color', 'clarity']
            numerical_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']

            ## Define the the custom ranking for each ordinal veriables (i.e order for categorical feactures)
            cut_categorie = ['Fair','Good','Very Good','Premium','Ideal']
            color_categories = ['D','E','F','G','H','I','J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            
            ## Pioelines
            ## Numericali Pipeline
            logging.info('Pipeline Initiated')
            num_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            ## categorical Pipeline
            cat_pipeline = Pipeline(
                steps =[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencode',OrdinalEncoder(categories=[cut_categorie,color_categories,clarity_categories])),
                    ('scaler',StandardScaler()) 
                ]
            )

            preprocessor=ColumnTransformer([
                ('nem_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
            ])
            logging.info('Pipeline completed')
            
            return preprocessor
        

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
        

    def initiate_data_Transforamtion(self,train_path,test_path):
        
        try:
            ## Reading the Train and Test data_sets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read Train and Test data completed')
            logging.info(f'Train DataFrame Head : \n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head : \n{test_df.head().to_string()}')

            logging.info('Obtaing the preprocessor object')
            preprocessing_obj = self.get_data_transformation_object()
            
            target_column = 'price'
            drop_columns = [target_column,'id']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column]
            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info('Applyting prepeocessing object on training and testing datasets.')

            ## Transforming using prepeocessor object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            ## creating numpy arrays to concatenate
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            ## calling the save_object in utils folder
            save_object(    
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info("preprocessor Pickel file saving completed.")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


            
        
        except Exception as e:
            logging.error('Exception occured in the initiate_data_transformation')
            raise CustomException(e,sys)