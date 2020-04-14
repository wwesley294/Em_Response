import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    df_messages = pd.read_csv(messages_filepath)
    # load categories dataset
    df_categories = pd.read_csv(categories_filepath)

    # create a dataframe of the 36 individual category columns
    df_categories = df_categories['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = df_categories.iloc[0]
    # extract a list of new column names for categories.
    category_colnames = [item[:-2] for item in row]
    # rename the columns of `categories`
    df_categories.columns = category_colnames

    # set each value to be the last character of the string
    for col in df_categories.columns:
        df_categories[col] = df_categories[col].str[-1]
        # convert column from string to numeric
        df_categories[col] = df_categories[col].astype(int)

    # drop the original categories column from `df`
    # df_categories = df_categories.drop(columns='categories')
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df_messages, df_categories], axis=1)

    return df


def clean_data(df):
    df = df.drop_duplicates(subset=['message'], keep=False)
    df = df.drop(columns=['id', 'original', 'genre'])
    df = df.dropna(axis=0, how='any')
    return df


def save_data(df, database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('comm', engine, index=False)


'''

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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

'''


def main():
    messages_filepath = 'disaster_messages.csv'
    categories_filepath = 'disaster_categories.csv'
    database_filepath = 'em_comm.db'

    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)

    print('Cleaned data saved to database!')


if __name__ == '__main__':
    main()