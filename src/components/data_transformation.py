import os
import sys
import re
import string
from dataclasses import dataclass

import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'tfidf_vectorizer.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.vectorizer = TfidfVectorizer(max_features=3000,ngram_range=(1,3))
        self.label_encoder = LabelEncoder()
        self.lemmatizer=WordNetLemmatizer()

    def clean_text(self, text):
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

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test data read successfully.")

            # Clean text
            train_df['text'] = train_df['text'].apply(self.clean_text)
            test_df['text'] = test_df['text'].apply(self.clean_text)

            # Label encode
            train_df['target'] = self.label_encoder.fit_transform(train_df['target'])
            test_df['target'] = self.label_encoder.transform(test_df['target'])

            # TF-IDF fit and transform
            X_train = self.vectorizer.fit_transform(train_df['text']).toarray()
            X_test = self.vectorizer.transform(test_df['text']).toarray()

            y_train = train_df['target'].values
            y_test = test_df['target'].values

            # Save the vectorizer object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=self.vectorizer
            )

            logging.info("TF-IDF vectorizer saved successfully.")

            return (
                np.c_[X_train, y_train],
                np.c_[X_test, y_test],
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
