import pandas as pd
import numpy as np


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Clean numerical columns
    df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")
    df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

    # Fill missing values
    text_columns = [
        "Subject",
        "Title",
        "Institution",
        "Learning Product",
        "Level",
        "Duration",
        "Gained Skills"
    ]

    for col in text_columns:
        df[col] = df[col].fillna("")

    df["Rate"] = df["Rate"].fillna(0)
    df["Reviews"] = df["Reviews"].fillna(0)

    # Log transformation for popularity
    df["Reviews_Log"] = np.log1p(df["Reviews"])

    return df