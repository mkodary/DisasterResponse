# Disaster Response

## Table of Contents
- [Summary](#summary)
- [Installation](#installation)
- [Instructions](#instructions)

### Summary <a name="summary"></a>

This project analyzes data coming from **Figure Eight** and builds a model to classify disaster related messages.
This model will then be used to categorize these events/messages so that it be can sent to an appropriate disaster relief agency.

#### Repo components

- Data:
    - disaster_messages.csv: This csv file contains disaster events that will be used for training the model.
    - disaster_categories.csv: This csv file labels/categorizes the messages found in *disaster_messages.csv*
- ETL:
    - process_data.py. *This file performs the following procedures*:
        - Extracts the data from the two csv files.
        - Merges the two sets together.
        - Performs simple cleanup.
        - Loads the cleaned data to database which would be created in the data folder.
- ML Pipeline:
    - train_classifier: *This file performs the following procedures.
        - Fetches the data from the db.
        - Splits the data into training and testing sets.
        - Build the Pipeline.
        - Uses GridSearchCV to find the optimal params for the model
        - Trains the data using the GridSearchCV
        - Evaluates the result using *classification_report* provided by *sklearn*
        - saves the model using pickle.
- Flask framework:
    - Provides a nice UI for the user to input messages that will be classified by the model saved.
        

### Installation <a name="installation"></a>

1. Make sure that python is installed by issuing this command on the terminal:<br/>
`python --version`
2. Also you could follow the steps for installing pip, if not installed, over [here][pip-install] 
3. (optional) create a separate python env. Here is how to create [virtual env][env-install]
4. Execute this command: _(This should download all necessary packages for running this project)_. <br/>
`pip install -r requirements.txt`

### Instructions <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database:<br/>
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves:<br/>
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [pip-install]: <https://pip.pypa.io/en/stable/installing/>
   [env-install]: <https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/>