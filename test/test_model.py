import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
import pytest
import os
from unittest.mock import patch
from src.ml.model import train_model, evaluate_model_slices, compute_model_metrics, inference, save_lr_model, \
    aggregate_performance_metrics


@pytest.fixture
def sample_data():
    X_train = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
    y_train = np.array([0, 1, 0])
    return X_train, y_train


def test_train_model(sample_data):
    X_train, y_train = sample_data
    model = train_model(X_train, y_train)
    assert isinstance(model, LogisticRegression)


def test_evaluate_model_slices(sample_data):
    X_train, y_train = sample_data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    categorical_features = [0, 1, 2]  # Assuming all features are categorical
    slice_report = evaluate_model_slices(model, X_train, y_train, categorical_features)
    assert isinstance(slice_report, str)


def test_compute_model_metrics():
    y_true = np.array([0, 1, 0, 0, 1])
    preds = np.array([0, 1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, preds)
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0


def test_inference(sample_data):
    X_train, y_train = sample_data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = inference(model, X_train)
    assert len(preds) == len(X_train)


if __name__ == "__main__":
    pytest.main()
