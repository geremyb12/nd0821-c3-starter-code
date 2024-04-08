import numpy as np
import json
import os
from joblib import dump
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained logistic regression model.
    """
    # Initialize and train a Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

from sklearn.metrics import classification_report

def evaluate_model_slices(model, X, y, categorical_features):
    """Evaluate the performance of the model on slices of categorical features.

    This function evaluates the performance of the trained machine learning model
    on slices of the data based on categorical features. It calculates precision,
    recall, and F1-score for each slice separately.

    Parameters:
    -----------
    model : object
        Trained machine learning model.
    X : np.array
        Features data.
    y : np.array
        Labels data.
    categorical_features : list
        List of categorical feature names.

    Returns:
    --------
    dict
        A dictionary containing classification report for each slice.
    """
    slice_report = {}

    for feature in categorical_features:
        feature_slices = np.unique(X[:, categorical_features.index(feature)])

        for slice_value in feature_slices:
            slice_indices = np.where(X[:, categorical_features.index(feature)] == slice_value)
            X_slice = X[slice_indices]
            y_slice = y[slice_indices]

            preds_slice = model.predict(X_slice)

            precision, recall, fbeta = compute_model_metrics(y_slice, preds_slice)

            slice_report[(feature, slice_value)] = {
                "precision": precision,
                "recall": recall,
                "f1-score": fbeta,
            }

    return str(slice_report)


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model :
        Trained logistic regression model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # Perform inference using the trained model
    preds = model.predict(X)
    return preds

def save_lr_model(lr_model, encoder, lb, slice_report):
    """ Save model/encoder/label binarizer to file.

    Inputs
    ------
    lr_model : sklearn.linear_model.LogisticRegression
        Trained model.
    encoder : sklearn.preprocessing.OneHotEncoder
        Fitted encoder for values of category features.
    lb : sklearn.preprocessing.LabelBinarizer
        Fitted label binarizer.
    Returns
    -------
    None
    """
    directory = 'src/model/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    dump(lr_model, os.path.join(directory, 'lr_model.joblib'))
    dump(encoder, os.path.join(directory, 'encoder.joblib'))
    dump(lb, os.path.join(directory, 'label_binarizer.joblib'))
    with open(os.path.join(directory, 'slice_report.json'), 'w') as f:
        json.dump(slice_report, f)