import sys
import pandas as pd 
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUTS: Filepaths of both .csv files which contains the data (disaster_categories.csv and disaster_messages.csv)

    OUTPUTS: df (merged dataframe)

    The function reads the message data from disaster_messages.csv and the related categories to each message from
    disaster_categories.csv and merge then both into a single dataframe df;
    Then, categories are split into columns and the last two digits (which have no relevant meaning) are removed;
    Categories values are converted to integers ( 0 == not related, 1 == related);
    The columns associated to each category are then concatenated into df;
    '''
    #read data from csv
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner')

    #split categories into columns and drop the last two characters
    categories = df['categories'].str.split(';',expand=True)
    categories.columns = categories.iloc[0,:].apply(lambda x: x[:-2])

    #convert categories values to integers
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].str.replace('2','1')

        categories[column] = categories[column].astype(int)

    #concatenate the categories columns into df
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df,categories], axis=1)

    return df    


def clean_data(df):
    '''
    INPUTS: df (dataframe to be cleaned)

    OUTPUTS: cleaned dataframe

    The function drops the duplicated data from the dataframe (based on the column 'message')
    '''
    return df.drop_duplicates(subset='message', keep ='first')


def save_data(df, database_filename):
    '''
    INPUTS: df( dataframe to be saved into the database), database_filename(database to save data into)

    OUTPUTS:

    The function saves the dataframe into a sqlite database

    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('message',engine,index=False) 


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