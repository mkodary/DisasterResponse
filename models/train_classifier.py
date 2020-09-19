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
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """This function loads data from a database. Database must contain a table called 'Message'

    :param database_filepath: relative path for the database.
    :type database_filepath: str
    :return: DataFrame, DataFrame
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Message', engine)
    X_df = df['message']
    Y_df = df.iloc[:, 4:]

    return X_df, Y_df


def tokenize(text):
    """Custom tokening function.

    :type text: str
    :return: list
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get list of all urls using regex
    text = re.sub(url_regex, "urlplaceholder", text.lower())

    # tokenize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # initiate lemmatizer
    tokens = [lemmatizer.lemmatize(word.strip()) for word in tokens]

    return tokens


def build_model():
    """This function creates a pipeline and adds it to GridSearchCV for parameter tuning.

    :return: GridSearchCV object.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4],
    }

    return GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)


def evaluate_model(model, X_test, Y_test, category_names):
    # predict on test data
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names, zero_division=1))


def save_model(model, model_filepath):
    """Saves models as a pickle file.

    :param model: models object.
    :type model_filepath: str
    :return: None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X_df, Y_df = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X_df.values, Y_df.values, test_size=0.2)

        print('Building models...')
        model = build_model()

        print('Training models...')
        model.fit(X_train, Y_train)

        print('Evaluating models...')
        evaluate_model(model, X_test, Y_test, Y_df.columns)

        print('Saving models...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained models saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the models to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
