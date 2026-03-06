"""
preprocessing.py
----------------
Reusable preprocessing pipeline for the Loan Approval Prediction project.

Usage:
    from src.preprocessing import load_and_preprocess
    X_train, X_test, y_train, y_test = load_and_preprocess("data/train.csv")
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV and drop the non-informative Loan_ID column."""
    df = pd.read_csv(filepath)
    df.drop(columns=["Loan_ID"], inplace=True)
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
    - Categorical columns → mode
    - LoanAmount (numerical) → mean
    """
    categorical_cols = ["Gender", "Married", "Dependents", "Self_Employed",
                        "Credit_History", "Loan_Amount_Term"]
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    df["LoanAmount"].fillna(df["LoanAmount"].mean(), inplace=True)
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical variables and drop redundant dummy columns
    to avoid the dummy variable trap.
    """
    df = pd.get_dummies(df)

    redundant_cols = [
        "Gender_Female", "Married_No", "Education_Not Graduate",
        "Self_Employed_No", "Loan_Status_N"
    ]
    df.drop(columns=redundant_cols, inplace=True)

    rename_map = {
        "Gender_Male": "Gender",
        "Married_Yes": "Married",
        "Education_Graduate": "Education",
        "Self_Employed_Yes": "Self_Employed",
        "Loan_Status_Y": "Loan_Status"
    }
    df.rename(columns=rename_map, inplace=True)
    return df


def remove_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where any feature falls outside the 1.5×IQR fence."""
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
    return df[mask]


def apply_sqrt_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Square root transform right-skewed numerical columns
    to bring distributions closer to normal.
    """
    for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]:
        df[col] = np.sqrt(df[col])
    return df


def load_and_preprocess(filepath: str, test_size: float = 0.2, random_state: int = 0):
    """
    Full preprocessing pipeline. Returns train/test splits ready for model training.

    Steps:
        1. Load & drop Loan_ID
        2. Impute missing values
        3. One-hot encode & clean dummies
        4. Remove outliers (IQR)
        5. Square root transform skewed features
        6. Split features / target
        7. Apply SMOTE (on training set only)
        8. MinMax scale features

    Returns:
        X_train, X_test, y_train, y_test — numpy arrays
    """
    df = load_data(filepath)
    df = impute_missing(df)
    df = encode_features(df)
    df = remove_outliers_iqr(df)
    df = apply_sqrt_transform(df)

    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]

    # Apply SMOTE only to training split to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
