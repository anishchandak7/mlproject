import sys
import os
from dataclasses import dataclass
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

# Modelling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

@dataclass
class ModelTrainerConfig:
    model_path = os.path.join("artifacts", "model.pkl")
    evaluation_report_path = os.path.join("artifacts", "Evaluation.xlsx")

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    

    def initiate_model_trainer(self, train_arr, test_arr):

        '''
        Trains the set of various models and selects best model 
        based on its r2_score and returns its r2_score.
        '''
        
        try:
            logging.info("Initiating Model Training.")

            X_train, X_test, y_train, y_test = (
                train_arr[:, :-1], test_arr[:,:-1], train_arr[:, -1], test_arr[:, -1]
            )
            logging.info("Splitted Train and Test array.")

            models= {
                        "Linear Regression": LinearRegression(),
                        "Lasso": Lasso(),
                        "Ridge": Ridge(),
                        "Support Vector Regressor" : SVR(),
                        "K-Neighbors Regressor": KNeighborsRegressor(),
                        "Decision Tree": DecisionTreeRegressor(),
                        "Random Forest Regressor": RandomForestRegressor(),
                        "XGBRegressor": XGBRegressor(), 
                        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                        "AdaBoost Regressor": AdaBoostRegressor()
                    }

            params = {
                "Linear Regression" : {},
                "Lasso" : {},
                "Ridge" : {},
                "Support Vector Regressor" : {},
                "K-Neighbors Regressor" : {},
                "Decision Tree" : {'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']},
                "Random Forest Regressor" : {'n_estimators': [8,16,32,64,128,256]},
                "XGBRegressor" : {'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]},
                "CatBoosting Regressor" : {'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]},
                "AdaBoost Regressor" :{'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]}
            }
            models_report:dict = evaluate_model(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,models=models, params=params)

            models_report_df = pd.DataFrame.from_dict(models_report, orient='index', columns=['Train Score', 'Test Score'])

            models_report_df = models_report_df.sort_values("Test Score", ascending=False)

            models_report_df.to_excel(self.model_trainer_config.evaluation_report_path)
            
            logging.info("Evaluation Report Prepared and Saved in artifacts folder.")

            # To get best model score from dataframe.
            best_model_score = models_report_df.head(1)['Test Score'].values[0]
            
            # To get best model name from dataframe.
            best_model_name = models_report_df.head(1).index.values[0]

            if best_model_score < 0.6: # If Selected model scored less than 60%.
                logging.info("Selected Model scored {0} which is less than 0.6".format(best_model_score))
                raise CustomException("No best model found")
            
            best_model = models[best_model_name] # Best Model Selection
            logging.info("{0} is the best model based on Model Evaluation.".format(best_model_name))

            save_object(file_path=self.model_trainer_config.model_path,
                        obj=best_model)
            logging.info("Model saved successfully.")
            
            return best_model_score

        except Exception as e:
            raise CustomException(e,sys)