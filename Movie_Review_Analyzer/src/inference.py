import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re
from utilities import Exception
from sklearn.metrics import accuracy_score
import os
import time


def load_data():
    try:
        test_data_path = os.environ['TEST_DATA_DIR']
        test = pd.read_csv(test_data_path)
        return test
    except Exception as ex:
        raise (f'Error in loading file: {ex}', str(ex))


def preprocess_data(test, preprocess_col):
    def preprocess_reviews(text):
        text = text.lower()
        text = re.sub("[^a-z0-9]", " ", text)
        text = re.sub("(\s)+", " ", text)
        text = text.strip()
        return text
    try:
        test[preprocess_col] = test[preprocess_col].apply(preprocess_reviews)
        return test
    except Exception as ex:
        raise (f'Error in preprocessing data: {ex}', str(ex))


def load_vectorizer(test, col):
    try:
        tfidf_path = os.environ['TFIDF_MODEL_DIR']
        tfidf = joblib.load(open(tfidf_path, 'rb'))
        test = pd.DataFrame(tfidf.transform(
            test[col]).toarray(), columns=tfidf.get_feature_names_out())
        return test
    except Exception as ex:
        raise (f'Error in transforming using tfidf vectorizer: {ex}', str(ex))


def make_predictions(test):
    try:
        lr_path = os.environ['LR_MODEL_DIR']
        model = joblib.load(open(lr_path, 'rb'))
        predictions = model.predict(test)
        return predictions
    except Exception as ex:
        raise (f'Error in making predictions: {ex}', str(ex))


def compute_metrics(predictions):
    try:
        test_data_labels_path = os.environ['TEST_DATA_LABELS']
        actuals = pd.read_csv(test_data_labels_path)
        actuals = np.where(actuals == 'positive', 1, 0)
        accuracy = accuracy_score(actuals, predictions)
        print(f'Accuracy is {accuracy*100}%.')
    except Exception as ex:
        raise (f'Error in computing metrics: {ex}', str(ex))


def orchestrate_pipeline():
    start_time = time.time()

    print('Loading data.')
    df = load_data()
    print('Loading data complete.')

    print('Preprocessing data.')
    x_test = preprocess_data(df, 'review')
    print('Preprocessing data complete.')

    print('Transforming using tfidf vectorizer.')
    x_test = load_vectorizer(x_test, 'review')
    print('Transforming using tfidf vectorizer complete.')

    print('Predicting using LR model.')
    predictions = make_predictions(x_test)
    print(f'{len(predictions)} Predictions generated.')

    compute_metrics_flag = os.environ['COMPUTE_METRICS']
    if compute_metrics_flag != 'No':
        print('Compute Metrics')
        compute_metrics(predictions)

    end_time = time.time()
    print(f'Inference pipeline took {end_time - start_time} seconds.')


if __name__ == '__main__':
    orchestrate_pipeline()
