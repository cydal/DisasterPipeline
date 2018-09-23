# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk


from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn import preprocessing

import re
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


from sklearn.externals import joblib
import sys


def load_data(database_filepath):
    """Load dataframe from sql db


    Parameters:
    database_filepath (string): Path to database file to export

    Returns:
    X (Dataframe): Feature Vector
    Y (Dataframe): Target Vector
    category_names (List): List of strings for column names for categories


    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    sql =  "SELECT * FROM message"

    df = pd.read_sql_query(sql, engine)


    X = df.message
    Y = df[df.columns.tolist()[4:]]

    '''
    ndf = df.groupby('genre').count()
    key = [x for x in ndf.index]
    '''

    return(X, Y, df.columns.tolist()[4:])



def tokenize(text):
    """Function returns after performing preprocessing steps on text including
    tolower, tokenization, stopwords removal and lemmatize

    Parameters:
    text (string): Refers to individual words passed in

    Returns:
    stemmed(string): Returns text with operations performed.


    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    stemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    return(stemmed)


def build_model():
    """Build Machine Learning Model

    Returns (model): Pipeline and gridsearch model
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
    ])

    
    parameters = {'clf__estimator__min_samples_split':[2, 4, 6],
                  'clf__estimator__max_depth': [2, 4]}

    #parameters = {'clf__estimator__min_samples_split':[2]}
    cv = GridSearchCV(pipeline, parameters)

    return(cv)




def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function returns the performance of test set for each category_names

    model (model): Model passed in for prediction
    X_test (dataframe): Test set input features to predict on
    Y_test (dataframe): Ground truth target values
    category_names (List): String list of target category names
    """

    print("Testing Performance")
    print(classification_report(Y_test, model.predict(X_test), target_names=category_names))

    #Todo cat names


def save_model(model, model_filepath):
    """Saves model passed in to specified filepath
    model (model): Model to save
    model_filepath (string): Location to save model
    """
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

        best_model = model.best_estimator_

        print('Evaluating model...')
        evaluate_model(best_model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(best_model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
