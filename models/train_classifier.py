import sys
import re
import pickle
import string
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM disaster", engine)
    df.drop(['original','genre','id'],inplace=True,axis=1)
    X = df.message.values
    Y = df[df.columns[1:]]
    return X,Y,Y.columns

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    lemmed = [WordNetLemmatizer().lemmatize(word) for word in words]
    lemmed = [WordNetLemmatizer().lemmatize(word, pos='v') for word in lemmed]
    stemmed = [PorterStemmer().stem(word) for word in lemmed]
    return stemmed

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

def build_model():
    pipeline = Pipeline([
    ('feature',FeatureUnion([
        ('text',Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize, max_df=0.5, 
                                    max_features=5000, ngram_range=(1, 2),
                                )),
        ('tfidf',TfidfTransformer())
    ])),
    ('word_count',WordCount())

    ])),
    ("clf", MultiOutputClassifier(RandomForestClassifier(min_samples_split=2, random_state=42)))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred=model.predict(X_test)
    results_dict = {}

    for pred, label, col in zip(y_pred.transpose(), Y_test.values.transpose(), category_names):
        #print(col)
        #print(classification_report(label, pred))
        results_dict[col] = classification_report(label, pred,output_dict=True)
    weighted_avg = {}
    for key in results_dict.keys():
        weighted_avg[key] = results_dict[key]['weighted avg']

    df_wavg = pd.DataFrame(weighted_avg).transpose()
    print(df_wavg.head())

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

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