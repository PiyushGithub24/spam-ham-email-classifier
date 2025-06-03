# utils.py is used for defining some common functions which is used in different module of the project

import os
import sys
import pandas as pd

# import dill
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

from sklearn.metrics import accuracy_score,precision_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train,X_test,y_test,models):
    try:
        model_list = []
        accuracy_score_list =[]
        precision_score_list=[]

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train) # Train model

            # Make predictions
            y_test_pred = model.predict(X_test)
    
            # Evaluate Test dataset
            accuracy_score_test=accuracy_score(y_test,y_test_pred)
            precision_score_test=precision_score(y_test,y_test_pred)

            model_list.append(list(models.keys())[i])
            accuracy_score_list.append(accuracy_score_test)
            precision_score_list.append(precision_score_test)

            performance_df=pd.DataFrame(list(zip(model_list, accuracy_score_list,precision_score_list)), columns=['Model Name', 'Accuracy Score','Precision Score']).sort_values(by=['Precision Score',"Accuracy Score"],ascending=False)
            return performance_df

    except Exception as e:
        raise CustomException(e, sys)
    

def clean_text(text):
        try:
            text = text.lower()                             # Step 1: Lowercase
            text = nltk.word_tokenize(text)                 # Step 2: Tokenization
    
            y = []
            for i in text:
                if i.isalnum():                             # Step 3: Remove punctuation and symbols
                    y.append(i)

            text = []
            for i in y:
                if i not in stopwords.words('english'):     # Step 4: Remove stopwords
                    text.append(lemmatizer.lemmatize(i))    # Step 5: Lemmatize each word

            return " ".join(text)

        except Exception as e:
            raise CustomException(e, sys)
        


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)