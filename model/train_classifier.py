# import libraries
import re
import nltk
import pickle
import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download(['punkt', 'wordnet'])


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get list of all urls using regex
    text = re.sub(url_regex, "urlplaceholder", text.lower())

    # tokenize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # initiate lemmatizer
    tokens = [lemmatizer.lemmatize(word.strip()) for word in tokens]

    return tokens


def build_model(load_model_from_file=None):

    if load_model_from_file is not None:
        return pickle.load(load_model_from_file)

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {'vect__max_df': [0.7, 1.0],
                  'tfidf__use_idf': [True, False],
                  'clf__estimator__min_samples_split': [2, 5, 10],
                  'clf__estimator__n_estimators': [10, 30],
                  'clf__estimator__max_features': ['auto', 'sqrt'],
                  'clf__estimator__min_samples_leaf': [1, 2, 4],
                  }

    return GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)


def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


def load_from_database(database_filepath):

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Message', engine)
    X_df = df['message']
    Y_df = df.iloc[:, 4:]

    return X_df, Y_df


def main():
    # todo validate inputs are correct.
    database_filepath, model_filename = sys.argv[1:]

    X_df, Y_df = load_from_database(database_filepath)

    X_train, X_test, y_train, y_test = train_test_split(X_df.values, Y_df.values)
    model = build_model()
    model.fit(X_train, y_train)

    # todo predict and evaluate model

    save_model(model, model_filename)


if __name__ == '__main__':
    main()
