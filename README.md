# Disaster Response Pipeline
 
## 1. Introduction
In this project, we will apply our skills to analyze disaster data from **Figure Eight** to build a model for an API that classifies disaster messages.

We have a data set containing real messages that were sent during disaster events. You will be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

Our project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## 2. Project Components

#### 1. Layout 
- ETL Pipeline Preparation 
- ML Pipeline Preparation
#### 2. ETL Pipeline
In a Python script, **process_data.py**, write a data cleaning pipeline that:

- Loads the **messages** and **categories** datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

#### 3. Machine Learning Pipeline
In a Python script, **train_classifier.py**, write a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

#### 4. Flask Web App
We are providing much of the flask web app for you, but feel free to add extra features depending on your knowledge of flask, html, css and javascript. For this part, you'll need to:

- Modify file paths for database and model as needed
- Add data visualizations using Plotly in the web app. One example is provided for you


<img src='Images/webapp.png'></img>
<br></br>

#### Visualisation 1 : 
Showing distribution of Message Genres
<img src='Images/vis1.png'></img>
<br></br>

#### Visualisation 2 : 
Showing proportion of Message by Category
<img src='Images/vis2.png'></img>
<br></br>