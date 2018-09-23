# Disaster Response Pipeline

This project includes a web app where an emergency worker can input a new message and get classification results on several categories.

Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

Prerequisites
Packages required to get the project working -

##### Pandas
##### Numpy
##### nltk
##### sklearn
##### sqlalchemy
##### Flask
##### plotly

### Instructions:

## ETL

To run ETL pipeline that cleans data and stores in database

The process_data.py file takes in three arguments, the two csv files
along with the location and name to store the database file. The two dataframes
from the two csv files are merged and cleaned and then outputed as a sql database
file.

> python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db


## Machine Learning

To run ML pipeline that trains classifier and saves

> python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl


The train_classifier.py file takes in two arguments, the database file output
from the previous step and the name and location to save the machine Learning
model. In this section, we load the data, perform some preprocessing steps like
tokenization, removal of stopwords and lemmatization. The data is fitted to a
machine learning model that has gone through a gridsearch of the best Parameters
from those searched. The model is tested using the test set with the performance
displayed and the final model is saved.

Run all the commands from the root directory of the project.

> python app/run.py

Go to http://0.0.0.0:3001/

### Acknowledgments

https://gist.github.com/PurpleBooth/109311bb0361f32d87a2
