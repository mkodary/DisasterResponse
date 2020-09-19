# import libraries
import pandas as pd
import sys
from sqlalchemy import create_engine


def load_data(message_filepath, category_filepath):
    """This function accepts two paths, and returns the merge of the two files.

    :param message_filepath: This is the path for the csv file containing the messages.
    :type message_filepath: str
    :param category_filepath: This is the path for the csv file containing the categories available.
    :type category_filepath: str
    :return: DataFrame
    """
    return pd.merge(
        pd.read_csv(message_filepath),
        pd.read_csv(category_filepath),
        on='id'
    )


def standardize_data(data):
    # todo this function could be further generalized instead of only standardizing 'related' column
    """This function standardizes the category columns. Only applied to related column.

    :type data: DateFrame
    :return: DataFrame
    """
    return data['related'].replace([2], 1)


def transform_categories(categories):
    """This function takes the categories columns and converts its values from string to binary values.

    :param categories:
    :return:
    """
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = categories[column] = pd.to_numeric(categories[column])

    return categories


def get_new_categories(categories):

    categories = categories.str.split(';', expand=True)

    # take everything except for the last two chars. i.e relevant-1 => relevant
    categories.columns = categories.loc[1].apply(lambda x: x[:len(x) - 2])

    return transform_categories(categories)


def remove_duplicated(data):
    print('Duplicated rows: {}'.format(data.duplicated().sum()))

    return data.drop_duplicates()


def preprocess_data(data):
    categories = get_new_categories(data.categories)

    # drop the original categories column from `df`
    data.drop(columns=['categories'], inplace=True)
    data = pd.concat([data, categories], axis=1)
    data = remove_duplicated(data)

    return standardize_data(data)


def save_to_database(data, save_filepath):
    engine = create_engine('sqlite:///{}'.format(save_filepath))
    data.to_sql('Message', engine, index=False, if_exists='replace')


def main():

    # todo check if arguments were sent.
    message_filepath, category_filepath, database_filepath = sys.argv[1:]
    df = load_data(message_filepath, category_filepath)
    df = preprocess_data(df)
    save_to_database(df, database_filepath)


if __name__ == '__main__':
    main()
