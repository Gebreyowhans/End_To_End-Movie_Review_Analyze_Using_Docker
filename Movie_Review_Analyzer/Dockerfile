FROM jupyter/scipy-notebook

WORKDIR /app

COPY . .

ENV LR_MODEL_DIR=model/logistic_model.pkl
ENV TFIDF_MODEL_DIR=model/tfidf_vectorizer.pkl
ENV TRAIN_DIR=data/imdb_dataset.csv
ENV TEST_DATA_DIR=data/test_set.csv
ENV TEST_DATA_LABELS=data/test_set_answers.csv
ENV COMPUTE_METRICS=No

RUN pip install -r requirements.txt 
USER root
RUN python src/train.py

RUN echo "Training model successful. Please run inference.py now to generate predictions."