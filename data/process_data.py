# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    """Outputs merged and loaded csv files"""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    #Merge datasets
    df = pd.merge(messages, categories, on="id")

    return(df)


def clean_data(df):
    """Outputs cleaned df"""

    #Preprocessing
    cols = df.categories.str.split(";", expand=True).iloc[0]

    # Split category and take first part as col names
    for i in range(len(cols)):
        cols[i] = cols[i].split("-")[0]

    category = df.categories.str.split(";", expand=True)
    category.columns = cols

    # Drop first part of categories and convert to int
    for col in cols:
        category[col] = pd.to_numeric(category[col].str.split("-").str[1])

    newdf = pd.concat([df, category], axis=1)
    newdf = newdf.drop(["categories"], axis=1)



    return(newdf)


def save_data(df, database_filename):
    """Saves cleaned data to DB"""

    #Create engine and save to sql db
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('message', engine, index=False)


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


if __name__ == '__main__':
    main()
