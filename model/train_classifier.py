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


def load_data(database_filepath):

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Message', engine)
    X_df = df['message']
    Y_df = df.iloc[:, 4:]

    return X_df, Y_df


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


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X_df, Y_df = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X_df.values, Y_df.values, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, Y_df.columns)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
