from sklearn.model_selection import train_test_split
from src.ml.data import process_data
from src.ml.model import train_model, evaluate_model_slices, save_lr_model, aggregate_performance_metrics
import pandas as pd


def train_and_save_model():
    """Loads data, trains a model, and saves it.

    This function loads the data, splits it into training and testing sets,
    processes the training data, trains a machine learning model, and saves
    the trained model.
    """
    # Load data
    data = pd.read_csv("./src/data/census.csv")
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20, stratify=data['salary'])

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    # Process the training & test data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", encoder=encoder, lb=lb, training=False
    )

    # Train the model
    model = train_model(X_train, y_train)

    slice_report = evaluate_model_slices(model, X_test, y_test, cat_features)

    aggregated_scores = aggregate_performance_metrics(slice_report)
    save_lr_model(model, encoder, lb, slice_report,aggregated_scores)


if __name__ == "__main__":
    train_and_save_model()