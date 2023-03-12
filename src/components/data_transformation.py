import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):

        try:

            num_features = ['writing_score', 'reading_score']
            cat_features = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            numerical_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            logging.info("Numerical features pipeline successfully created.")

            categorical_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("oh_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical features pipeline successfully created.")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", numerical_pipeline, num_features),
                    ("cat_pipeline", categorical_pipeline, cat_features)
                ]
                )
            
            logging.info("Preprocessor built successfully.")
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_data_path, test_data_path):

        try:
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            logging.info('Train and Test Read Successfully.')
            logging.info('Obtaining preprocessor Object.')

            preprocessor = self.get_data_transformer_object()

            target_col_name = 'math_score'
            numerical_features = ['writing_score', 'reading_score']

            X_train = train_data.drop(columns=[target_col_name], axis=1)
            y_train = train_data[target_col_name]

            X_test = test_data.drop(columns=[target_col_name], axis=1)
            y_test =test_data[target_col_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessor)

            logging.info('Saved Preprocessor object.')

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            raise CustomException(e, sys)
    

