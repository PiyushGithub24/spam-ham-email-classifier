import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object,clean_text


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,email_content):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','tfidf_vectorizer.pkl')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            email_content=clean_text(str(email_content))                            #data cleaning
            data_vectorized = preprocessor.transform([email_content]).toarray()    #vectorization
            preds=model.predict(data_vectorized)                               #prediction
            return int(preds[0])
        
        except Exception as e:
            raise CustomException(e,sys)
        

# pp=PredictPipeline()
# print(pp.predict("This is a reminder that we have a meeting scheduled for tomorrow at 2:00 PM regarding Project Alpha. Please join us at the conference room on the 3rd floor. "))
