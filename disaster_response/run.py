import json
import plotly
import pandas as pd
import re

import nltk
nltk.download(['punkt', 'stopwords','wordnet'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

import tokenize_function as tkf

app = Flask(__name__, template_folder='template')



@app.before_first_request

def model_get_data():

    global df
    global model

    #load data
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df = pd.read_sql_table('message', engine)

    # load model
    model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    def truncate(n):
        return int(n * 1000) / 1000

    cat_counts = []
    for column in df.columns[4:]:
        cat_percent =((df[column].sum()/df.shape[0])*100)
        cat_counts.append(truncate(cat_percent))

    def remove_underline(text):
        text = re.sub(r"[^a-zA-Z0-9]"," ",text)
        return text

    cat_names = []
    for i in range(len(df.columns[4:])):
        clean_name = remove_underline(df.columns[4:][i])
        cat_names.append(clean_name)

        
    # create visuals
    
        
    graphs = [
        # graph two - Distribution of Categories
        {
            'data': [
                Bar(
                    x =cat_names ,
                    y =cat_counts ,
                )
            ],

            'layout' : {
                'title' : 'Distribution of Categories',
                'yaxis': {
                    'title': "Count [%]"
                },
                'xaxis': {
                    'title': " "
                }
                
            }
        },

        # graph one - Distribution on Message Genres
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }

    ]
  
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

if __name__ == '__main__':
    app.run()
