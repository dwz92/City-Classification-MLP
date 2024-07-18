'''
Author: Emma Teng
Contact: tengemma136@gmail.com
'''

import utils
import MLPModel
import numpy as np

def predict(x):
    """
    Helper function to make prediction for a given input x.
    This code is here for demonstration purposes only.
    """
    # randomly choose between the four choices: 'Dubai', 'Rio de Janeiro', 'New York City' and 'Paris'.
    # NOTE: make sure to be *very* careful of the spelling/capitalization of the cities!!

    model = MLPModel(num_features=71, num_hidden= 208, num_classes=4)

    y = model.forward(x)
    return y


def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    # read the file containing the test data
    # you do not need to use the "csv" package like we are using
    # (e.g. you may use numpy, pandas, etc)
    data = utils.preprocess_data_for_prediction(filename, catagories=["Q1", "Q2", "Q3", "Q4", "Q7", "Q5", "Q6"])

    predictions = []
    for index, test_example in data.iterrows():
        # obtain a prediction for this test example
        pred = predict(test_example)
        res = np.eye(4)[np.argmax(pred)].astype(np.int64)
        city = ''

        if np.where(res==1)[0][0] == 0:
            city = 'Dubai'
        elif np.where(res==1)[0][0] == 1:
            city = 'Rio de Janeiro'
        elif np.where(res==1)[0][0] == 2:
            city = 'New York City'
        else:
            city = 'Paris'

        predictions.append(city)

    return predictions