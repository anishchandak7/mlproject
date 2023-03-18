import os
import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object
from src.logger import logging


class PredictPipeline:
    def __init__(self) -> None:
        self.model_path = 'artifacts\model.pkl'
        self.preprocessor_path = 'artifacts\preprocessor.pkl'

    def predict(self, features):
        
        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)
            logging.info('PredictionPipeline - Model and preprocessor are successfully loaded.')
            transformed_features = preprocessor.transform(features)
            logging.info('PredictionPipeline - Features are scaled and transformed.')
            prediction = model.predict(transformed_features)
            logging.info('PredictionPipeline - Prediction Done!! {0}'.format(prediction))
            return prediction
        except Exception as e:
            raise CustomException(e,sys)

class DataWrapper:

    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    

    def to_dataframe(self) -> pd.DataFrame:

        '''
        Transforms the features into dataframe.
        '''

        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)