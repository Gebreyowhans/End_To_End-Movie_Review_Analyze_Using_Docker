import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re
from utilities import Exception
import os
import time


def load_data():
    try:
        path = os.environ['TRAIN_DIR']
        imdb_reviews_df = pd.read_csv(path)
        return imdb_reviews_df
    except Exception as ex:
        raise (f'Error in loading file: {ex}', str(ex))


def split_data(df, target_col):
    try:
        x_train, x_test1, y_train, y_test1 = train_test_split(df.drop(
            target_col, axis=1), df[target_col], stratify=df[target_col], test_size=0.25, random_state=60)
        x_test1, _, y_test1, _ = train_test_split(
            x_test1, y_test1, stratify=y_test1, test_size=0.5, random_state=60)

        test_data_path = os.environ['TEST_DATA_DIR']
        test_data_labels_path = os.environ['TEST_DATA_LABELS']

        x_test1.to_csv(test_data_path, index=False)
        y_test1.to_csv(test_data_labels_path, index=False)
        return (x_train, y_train)
    except Exception as ex:
        raise (f'Error in splitting data: {ex}', str(ex))


def map_target_label(target_series):
    try:
        mapper = {'positive': 1, 'negative': 0}
        target_series = target_series.map(mapper)
        return target_series
    except Exception as ex:
        raise (f'Error in mapping target label: {ex}', str(ex))


def preprocess_data(train, preprocess_col):
    def preprocess_reviews(text):
        text = text.lower()
        text = re.sub("[^a-z0-9]", " ", text)
        text = re.sub("(\s)+", " ", text)
        text = text.strip()
        return text
    try:
        train[preprocess_col] = train[preprocess_col].apply(preprocess_reviews)
        return train
    except Exception as ex:
        raise (f'Error in preprocessing data: {ex}', str(ex))


def train_and_save_vectorizer(train, target_col):
    try:
        tfidf = TfidfVectorizer(max_features=200, stop_words='english')
        tfidf.fit(train[target_col])
        tfidf_path = os.environ['TFIDF_MODEL_DIR']
        joblib.dump(tfidf, open(tfidf_path, 'wb'), compress=True)
        train = pd.DataFrame(tfidf.transform(
            train[target_col]).toarray(), columns=tfidf.get_feature_names_out())
        return train
    except Exception as ex:
        raise (f'Error in training and saving tfidf vectorizer: {ex}', str(ex))


def train_model(train, labels):
    try:
        model = LogisticRegression(random_state=42)
        model.fit(train, labels)
        lr_path = os.environ['LR_MODEL_DIR']
        joblib.dump(model, open(lr_path, 'wb'))
    except Exception as ex:
        raise (f'Error in training and saving model: {ex}', str(ex))


def orchestrate_pipeline():
    start_time = time.time()

    print('Loading data.')
    df = load_data()
    print('Loading data complete.')

    print('Splitting data.')
    x_train, y_train = split_data(df, 'sentiment')
    print('Splitting data complete.')

    print('Mapping target labels.')
    y_train = map_target_label(y_train)
    print('Mapping target labels complete.')

    print('Preprocessing data.')
    x_train = preprocess_data(x_train, 'review')
    print('Preprocessing data complete.')

    print('Training tfidf vectorizer.')
    x_train = train_and_save_vectorizer(x_train, 'review')
    print('Training tfidf vectorizer complete.')

    print('Training Logistic Regression model.')
    train_model(x_train, y_train)
    print('Training Logistic Regression model complete.')

    end_time = time.time()
    print(f'Train pipeline took {end_time - start_time} seconds.')


if __name__ == '__main__':
    orchestrate_pipeline()
