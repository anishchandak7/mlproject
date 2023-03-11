import sys
import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts","train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")
    raw_data_path = os.path.join("artifacts", "data.csv")


SOURCE_PATH = 'notebook\data\stud.csv'
class DataIngestion:

    def __init__(self):
        logging.info("Data Ingestion Configuration Setup Initiated.")
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        try:
            logging.info("Data Ingestion Started.")
            
            # Reading from source path.
            df = pd.read_csv(SOURCE_PATH)
            logging.info("Dataset has been read successfully.")

            # Setting up artifacts folder.
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            logging.info("'artifact' folder has been created successfully if not exist before.")

            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)
            logging.info('Successfully saved Raw data in artifacts folder.')

            logging.info("Train Test Split Initiated.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42) # 80% Training Data 20% Test Data.

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Train and Test data has been successfully saved in artifacts folder.')
            logging.info('Data Ingestion is completed.')

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()