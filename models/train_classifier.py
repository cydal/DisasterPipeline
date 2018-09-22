# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('stopwords')


from nltk.stem.porter import PorterStemmer
from sklearn import preprocessing

import re
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import GaussianNB


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


from sklearn.externals import joblib
import sys


def load_data(database_filepath):
    """Load dataframe from sql db"""
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    sql =  "SELECT * FROM message"

    df = pd.read_sql_query(sql, engine)

    le.fit(df[df.columns.tolist()[3]])

    X = df[[x for x in df.columns.tolist() if x not in ["id", "genre", "original"]]]
    Y = df[df.columns.tolist()[4:]]

    #Y = le.transform(df[df.columns.tolist()[3]])

    '''
    ndf = df.groupby('genre').count()
    key = [x for x in ndf.index]
    '''

    return(X, Y, df.columns.tolist()[4:])



def tokenize(text):
    """Function returns after performing preprocessing steps on text including
    tolower, tokenization, stopwords removal and stemming"""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    stemmed = [PorterStemmer().stem(w) for w in words]
    return(stemmed)


def build_model():
    """Build Machine Learning Model"""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', ExtraTreesClassifier())
    ])

    return(pipeline)




def evaluate_model(model, X_test, Y_test, category_names):
    """"""

    print("Testing Performance")
    print(classification_report(Y_test, pipeline.predict(X_test.message)))

    #Todo cat names


def save_model(model, model_filepath):
    """Saves model passed in to specified filepath"""
    joblib.dump(model, model_filepath)


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
