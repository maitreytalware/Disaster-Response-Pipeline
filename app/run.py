import nltk
nltk.download('stopwords')
import json
import plotly
import pandas as pd
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from collections import Counter
import json, plotly
import numpy as np
import operator
from pprint import pprint
import re
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
app = Flask(__name__)


class WordCount(BaseEstimator, TransformerMixin):
    def word_count(self, text):
        table = text.maketrans(dict.fromkeys(string.punctuation))
        words = word_tokenize(text.lower().strip().translate(table))
        return len(words)
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        count = pd.Series(x).apply(self.word_count)
        return pd.DataFrame(count)


def tokenize(text):
    """
    Tokenizes text data
    Args:
    text str: Messages as text data
    Returns:
    # clean_tokens list: Processed text after normalizing, tokenizing and lemmatizing
    words list: Processed text after normalizing, tokenizing and lemmatizing
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    words = word_tokenize(text)
    stopwords_ = stopwords.words("english")
    words = [word for word in words if word not in stopwords_]
    
    # extract root form of words
    words = [WordNetLemmatizer().lemmatize(word) for word in words]

    return words

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    genre_counts = df.groupby('genre').count()['message']
                                                          
    genre_names = list(genre_counts.index)                
    cat_p = df[df.columns[4:]].sum()/len(df)              
                                                          
    cat_p = cat_p.sort_values(ascending = False)          
                                                         
    cats = list(cat_p.index)                              

    words_with_repetition=[]                              
                                                          
                                                          
    figures = [
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
                    x=cats,
                    y=cat_p
                )
            ],

            'layout': {
                'title': 'Proportion of Messages <br> by Category',
                'yaxis': {
                    'title': "Proportion",
                    'automargin':True
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -40,
                    'automargin':True
                }
            }
        }

    ]
    

    ids = ["figure-{}".format(i) for i, _ in enumerate(figures)]
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html', ids=ids, figuresJSON=figuresJSON, data_set=df)


@app.route('/go')
def go():

    query = request.args.get('query', '') 

    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))


    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()