# import libraries
import re
import nltk
import joblib
import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """ Our custom Transfomer, checks if sentence starts with a verb.
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    """This function loads data from a database. Database must contain a table called 'Message'

    :param database_filepath: relative path for the database.
    :type database_filepath: str
    :return:
        - X_df: DataFrame
        - Y_df: DataFrame
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
    :return:
        - tokens: list
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

    :return:
        - GridSearchCV object.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = get_tune_params()

    return GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=5)


def get_tune_params():
    """Function that creates the param dict.

    :return: Dict
    """
    return {
        'features__text_pipeline__vect__max_df': (0.75, 1.0),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10, 30],
        'clf__estimator__min_samples_split': [2, 3, 4],
    }


def evaluate_model(model, X_test, Y_test, category_names):
    """This function evaluates the model by predicting the X_test and generating a report using classification_report

    :param model: The model to use for predicting
    :type X_test: DataFrame
    :param Y_test: expected results when predicting
    :param category_names: the labels provided.
    :return:
        - bool or dict: Bool if failed to create a report, else returns a dict
    """
    # predict on test data
    y_pred = model.predict(X_test)

    return classification_report(Y_test, y_pred, target_names=category_names, zero_division=1)


def save_model(model, model_filepath):
    """Saves models as a pickle file.

    :param model: models object.
    :type model_filepath: str
    :return: None
    """
    joblib.dump(model.best_estimator_, open(model_filepath, 'wb'), compress=1)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X_df, Y_df = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X_df.values, Y_df.values, test_size=0.2)

        print('Building models...')
        model = build_model()

        print('Training models...')
        model = model.fit(X_train, Y_train)

        print('Evaluating models...')
        report = evaluate_model(model, X_test, Y_test, Y_df.columns.values.tolist())
        print (report)

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
