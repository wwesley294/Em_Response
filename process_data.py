import sys
import numpy as np
import pandas as pd
import matplotlib as plt
from sqlalchemy import create_engine


'''

process_data is a data ETL pipeline for message and category data. It prepares the data for the ML pipeline in the next step.

'''

# load_data takes two arguments of file path and return a concatenated DataFrame.
def load_data(messages_filepath, categories_filepath):
    
    # Load message and category data.
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)
    
    # Expand the columns to house individual categories and create new column names.
    df_categories = df_categories['categories'].str.split(';', expand=True)
    row = df_categories.iloc[0]
    category_colnames = [item[:-2] for item in row]
    df_categories.columns = category_colnames
    
    # Convert the new column values into 1 and 0.
    for col in df_categories.columns:
        df_categories[col] = df_categories[col].str[-1]
        df_categories[col] = df_categories[col].astype(int)
    
    # Concatenate the message DataFrame and new category DataFrame.
    df = pd.concat([df_messages, df_categories], axis=1)
    
    return df

# clean_data takes a DataFrame as argument, filters unnecessary data, and returns a clean DataFrame.
def clean_data(df):

    df = df.drop_duplicates(subset=['message'], keep=False)
    df = df.drop(columns=['id','original'])
    df = df.dropna(axis=0, how='any')

    return df

# save_data takes a DataFrame and file path as arguments.
# Store data in a new database and table.
def save_data(df, database_filepath):

    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('comm', engine, index=False)

# main provides file paths and executes the functions above in order.
def main():

    messages_filepath = '/home/workspace/data/disaster_messages.csv'
    categories_filepath = '/home/workspace/data/disaster_categories.csv'
    database_filepath = '/home/workspace/models/em_comm.db'

    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)
        
    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)
        
    print('Cleaned data saved to database!')   


if __name__ == '__main__':
    main()