import os
import sys
from src.exception import CustomException
from sklearn.metrics import r2_score
import dill

def save_object(file_path, obj):

    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, X_test, y_train, y_test, models)-> dict:

    try:
        models_report = {}

        for model_name in models:

            model = models[model_name]

            model.fit(X_train, y_train) # Model Training

            # Model Prediction
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Model Evaluation.
            model_train_score = r2_score(y_train_pred, y_train)
            model_test_score = r2_score(y_test_pred, y_test)

            models_report[model_name] = [model_train_score, model_test_score]

        return models_report
    
    except Exception as e:
        raise CustomException(e, sys)


