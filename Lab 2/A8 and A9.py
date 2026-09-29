import pandas as pd
import numpy as np


def load_dataset(file_path):
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")  # load dataset


def preprocess_dataset(df):
    df = df.replace("?", np.nan).infer_objects(copy=False)  # replace ? with NaN

    numeric_columns = [
        "TSH",
        "T3",
        "TT4",
        "T4U",
        "FTI",
        "TBG"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column])  # convert to numeric

    return df


def has_outliers(column):
    q1 = column.quantile(0.25) # q 1
    q3 = column.quantile(0.75) # q 3

    iqr = q3 - q1 # inter quantile range

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    return ((column < lower_bound) | (column > upper_bound)).any() # return any outside of range


def impute_missing_values(df):

    numeric_columns = df.select_dtypes(include=np.number).columns
    categorical_columns = df.select_dtypes(exclude=np.number).columns

    for column in numeric_columns: # fill numeric columns

        if has_outliers(df[column]):
            value = df[column].median()  # median for outliers

        else:
            value = df[column].mean()  # mean otherwise

        df[column] = df[column].fillna(value)

    for column in categorical_columns: # fill categorical columns

        value = df[column].mode()[0]  # most frequent value

        df[column] = df[column].fillna(value)

    return df

def normalize_data(df):

    numeric_columns = [
        "age",
        "TSH",
        "T3",
        "TT4",
        "T4U",
        "FTI",
        "TBG"
    ]

    for column in numeric_columns:

        minimum = df[column].min()
        maximum = df[column].max()

        df[column] = (df[column] - minimum) / (maximum - minimum)

    return df


def main():

    df = load_dataset("Lab Session Data.xlsx")

    df = preprocess_dataset(df)

    df = impute_missing_values(df)

    print("Missing Values After Imputation:\n")
    print(df.isna().sum())

    df = normalize_data(df)

    print("\nNormalized Data:\n")

    print(df[[
        "age",
        "TSH",
        "T3",
        "TT4",
        "T4U",
        "FTI",
        "TBG"
    ]].head())

if __name__ == "__main__":
    main()