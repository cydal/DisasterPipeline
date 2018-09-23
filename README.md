# Disaster Response Pipeline

This project includes a web app where an emergency worker can input a new message and get classification results on several categories.

Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

Prerequisites
Packages required to get the project working - 

Pandas
Numpy
nltk
sklearn
sqlalchemy
Flask
plotly

Instructions:

ETL

To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db


Machine Learning

To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

Run the following command in the app's directory to run your web app. 

> python run.py

Go to http://0.0.0.0:3001/

### Acknowledgments

https://gist.github.com/PurpleBooth/109311bb0361f32d87a2