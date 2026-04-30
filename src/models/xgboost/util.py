# utils.py

import pandas as pd


def load_data():
    """Load training and validation data"""
    X_train = pd.read_csv("data/processed/v1/X_train.csv")
    y_train = pd.read_csv("data/processed/v1/y_train.csv")

    X_val = pd.read_csv("data/processed/v1/X_val.csv")
    y_val = pd.read_csv("data/processed/v1/y_val.csv")

    return X_train, y_train, X_val, y_val


def load_test_data():
    """Load test data"""
    X_test = pd.read_csv("data/processed/v1/X_test.csv")
    y_test = pd.read_csv("data/processed/v1/y_test.csv")

    return X_test, y_test