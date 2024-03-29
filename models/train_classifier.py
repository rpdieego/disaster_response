import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC

import tokenize_function as tkf


def load_data(database_filepath):
    '''
    INPUT: database_filepath (path to access the database which holds the cleaned data)

    OUTPUT: X (estimators), y (target variable), category_names (list of strings)

    The function reads the cleaned data from the sqlite database, and split it into the estimators (X), target variables (y)
    and category_names to be used displaying the results of the model;

    X = Messages
    y = Categories binarized

    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('message',engine)
    X = df.message.values
    Y = df.iloc[:,4:].values
    category_names = df.columns[4:]

    return X, Y, category_names

def build_model():
    '''
    INPUT: none

    OUTPUT: model

    Function builds the model as a machile learning learning pipeline
    '''

    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tkf.tokenize, ngram_range = (1,2))),
        ('tfidf', TfidfTransformer()),
        ('clf',MultiOutputClassifier(OneVsRestClassifier(LinearSVC(), n_jobs=1)))  
                        ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),    
                 }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1, cv=5)
                        
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT: model(built mode), X_test (estimators test set), Y_test(target test set), category_names (category labels)

    OUTPUT: none

    Function generates the model predictons (Y_hat) to the estimators test set (X_test) and print the classification report
    '''
    Y_hat = model.predict(X_test)
    print(classification_report(Y_test,Y_hat, target_names = category_names))



def save_model(model, model_filepath):
    '''
    INPUT: model (built model), model_filepath (path to save the model as a pickle file)

    OUTPUT:none

    Function saves the built mode as a picke file at model_filepath  

    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=17, shuffle=True)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()