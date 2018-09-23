import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine
import re

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer


app = Flask(__name__)

def tokenize(text):
    """Function returns after performing preprocessing steps on text including
    tolower, tokenization, stopwords removal and stemming"""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    stemmed = [PorterStemmer().stem(w) for w in words]
    return(stemmed)


# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('message', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)


    ### Visualization setup for percentage of overall messages
    ### for categories


    total_messages = df.shape[0]
    category_names = df.columns.values[4:]
    percentage_values = [(sum(df[cat]) * 100/total_messages) for cat in category_names]


    ### Visualization setup for typical length of messages

    df["msglength"] = df.message.str.len()
    grouped = df.groupby('genre').mean()["msglength"]


    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
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
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=percentage_values
                )
            ],

            'layout': {
                'title': 'Percentage of Messages per Category',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=grouped.index,
                    y=grouped.values
                )
            ],
            'layout': {
                'title': 'Average of Message Length Per Genre',
                'yaxis': {
                    'title': "Average"
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


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
