import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import accuracy_score,precision_score

from src.utils import save_object,evaluate_models
import pandas as pd

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Support Vector Classifier" : SVC(kernel='sigmoid', gamma=1.0),
                "K-Neighbors Classifier" : KNeighborsClassifier(),
                "Multinomial Naive Bayes'" : MultinomialNB(),
                "Decision Tree Classifier" : DecisionTreeClassifier(max_depth=5),
                "Logistic Regression" : LogisticRegression(solver='liblinear', penalty='l1'),
                "RandomForest Classifier" :RandomForestClassifier(n_estimators=50, random_state=2),
                "Bagging Classifier" : BaggingClassifier(n_estimators=50, random_state=2),
                "Gradient Boosting Classifier" : GradientBoostingClassifier(n_estimators=50,random_state=2)
            }


            model_report:pd.DataFrame=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                            models=models)
            
            ## To get best model accuracy score from Dataframe of evaluated models
            best_model_score = model_report.sort_values(by=["Accuracy Score", "Precision Score"], ascending=False).iloc[0]['Accuracy Score']

            ## To get best model name from Dataframe of evaluated models

            best_model_name = model_report.sort_values(by=["Accuracy Score", "Precision Score"], ascending=False).iloc[0]['Model Name']

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy

        except Exception as e:
            raise CustomException(e,sys)