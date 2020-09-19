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


def transform_categories(categories):
    """This function takes the categories columns and converts its values from string to binary values.

    :type categories: DataFrame
    :return: DataFrame
    """
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = categories[column] = pd.to_numeric(categories[column])

    return categories


def get_new_categories(categories):
    """

    :type categories: DataFrame
    :return: DataFrame
    """

    categories = categories.str.split(';', expand=True)

    # take everything except for the last two chars. i.e relevant-1 => relevant
    categories.columns = categories.loc[1].apply(lambda x: x[:len(x) - 2])

    return transform_categories(categories)


def remove_duplicated(data):
    print('Duplicated rows: {}'.format(data.duplicated().sum()))

    return data.drop_duplicates()


def clean_data(data):
    """

    :type data: DataFrame
    :return: DataFrame
    """
    categories = get_new_categories(data.categories)

    # drop the original categories column from `df`
    data.drop(columns=['categories'], inplace=True)
    data = pd.concat([data, categories], axis=1)
    data = remove_duplicated(data)

    data['related'] = data['related'].replace([2], 1)

    # todo transform genre into dummies.

    return data


def save_data(data, save_filepath):
    """

    :type data: DataFrame
    :type save_filepath: str
    :return: None
    """
    engine = create_engine('sqlite:///{}'.format(save_filepath))
    data.to_sql('Message', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
