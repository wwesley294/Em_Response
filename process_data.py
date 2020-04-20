import numpy as np
import pandas as pd
from sqlalchemy import create_engine


'''
process_data.py is a data ETL pipeline that prepares the message and category data for the ML pipeline.

'''

# load_data takes two arguments to file paths to create a new DataFrame as a single source of data.
def load_data(messages_filepath, categories_filepath):

    # Load messages and categories from CSVs into DataFrames.
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)

    # Split the categories at ';' and expand them into 36 individual columns.
    df_categories = df_categories['categories'].str.split(';', expand=True)

    # Reconstruction column name and apply to the new DataFrame.
    row = df_categories.iloc[0]
    category_colnames = [item[:-2] for item in row]
    df_categories.columns = category_colnames

    # Convert category columns into binary data of integers 0 and 1.
    for col in df_categories.columns:
        df_categories[col] = df_categories[col].str[-1]
        df_categories[col] = df_categories[col].astype(int)

    # Concatenate the two DateFrames.
    df = pd.concat([df_messages, df_categories], axis=1)

    return df


# clean_data takes an argument of the new DataFrame and further removed unnecessary data.
def clean_data(df):

    # Remove rows with duplicate message and null values. Also drop the "id" and "original" columns.
    df = df.drop_duplicates(subset=['message'], keep=False)
    df = df.drop(columns=['id', 'original'])
    df = df.dropna(axis=0, how='any')

    return df


# save_data takes two arguments of the new DataFrame and file path. Deposits the new DataFrame to a database.
def save_data(df, database_filepath):

    # Create a SQLite database and save the DataFrame as table "comm".
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('comm', engine, index=False)


# main executes the functions above in order while providing progress updates.
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


# Execute main.
if __name__ == '__main__':
    main()
