import pandas as pd
import numpy as np


def load_dataset(file_path):
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")  # load dataset


def preprocess_dataset(df):
    df = df.replace("?", np.nan)  # replace with nan

    numeric_columns = [
        "TSH",
        "T3",
        "TT4",
        "T4U",
        "FTI",
        "TBG"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column]) # make them numeric instead of string

    return df


def identify_attribute_types(df):
    attribute_types = {}

    numeric_columns = [ 
        "Record ID",
        "age",
        "TSH",
        "T3",
        "TT4",
        "T4U",
        "FTI",
        "TBG"
    ]

    binary_columns = [
        "on thyroxine",
        "query on thyroxine",
        "on antithyroid medication",
        "sick",
        "pregnant",
        "thyroid surgery",
        "I131 treatment",
        "query hypothyroid",
        "query hyperthyroid",
        "lithium",
        "goitre",
        "tumor",
        "hypopituitary",
        "psych",
        "TSH measured",
        "T3 measured",
        "TT4 measured",
        "T4U measured",
        "FTI measured",
        "TBG measured"
    ]

    nominal_columns = [
        "sex",
        "referral source",
        "Condition"
    ]

    for column in df.columns:

        if column in numeric_columns:
            attribute_types[column] = "Numeric" # assigning the datatype for the attributes

        elif column in binary_columns:
            attribute_types[column] = "Binary"

        elif column in nominal_columns:
            attribute_types[column] = "Nominal"

        else:
            attribute_types[column] = "Unknown"

    return attribute_types


def suggest_encoding(df):
    encoding = {}

    binary_columns = [
        "on thyroxine",
        "query on thyroxine",
        "on antithyroid medication",
        "sick",
        "pregnant",
        "thyroid surgery",
        "I131 treatment",
        "query hypothyroid",
        "query hyperthyroid",
        "lithium",
        "goitre",
        "tumor",
        "hypopituitary",
        "psych",
        "TSH measured",
        "T3 measured",
        "TT4 measured",
        "T4U measured",
        "FTI measured",
        "TBG measured"
    ]

    nominal_columns = [
        "sex",
        "referral source",
        "Condition"
    ]

    for column in df.columns: # assign encoding techniques for different attributes

        if column in binary_columns: 
            encoding[column] = "Label Encoding"

        elif column in nominal_columns:
            encoding[column] = "One-Hot Encoding"

        else:
            encoding[column] = "Not Required"

    return encoding


def calculate_numeric_ranges(df):
    numeric_df = df.select_dtypes(include=np.number) # only for numeric columns

    ranges = {}

    for column in numeric_df.columns:
        ranges[column] = (
            numeric_df[column].min(),
            numeric_df[column].max()
        )

    return ranges


def calculate_missing_values(df):
    missing_values = {}

    for column in df.columns:
        missing_values[column] = df[column].isna().sum()  # count missing values

    return missing_values


def detect_outliers(df):
    numeric_df = df.select_dtypes(include=np.number) # only numeric values

    outliers = {}

    for column in numeric_df.columns:

        q1 = numeric_df[column].quantile(0.25) # lower quantile
        q3 = numeric_df[column].quantile(0.75) # upper quantile

        iqr = q3 - q1 # .5

        lower_bound = q1 - (1.5 * iqr) # lower quantile - .75
        upper_bound = q3 + (1.5 * iqr) # upper quantile + .75

        outlier_count = numeric_df[
            (numeric_df[column] < lower_bound) |
            (numeric_df[column] > upper_bound)
        ].shape[0] # every val outside of that range

        outliers[column] = outlier_count

    return outliers


def calculate_statistics(df):
    numeric_df = df.select_dtypes(include=np.number)

    statistics = {}

    for column in numeric_df.columns:

        statistics[column] = {
            "Mean": numeric_df[column].mean(),
            "Variance": numeric_df[column].var(),
            "Standard Deviation": numeric_df[column].std()
        } # use built in functions

    return statistics


def main():

    df = load_dataset("Lab Session Data.xlsx")

    df = preprocess_dataset(df)

    attribute_types = identify_attribute_types(df)

    encoding = suggest_encoding(df)

    ranges = calculate_numeric_ranges(df)

    missing_values = calculate_missing_values(df)

    outliers = detect_outliers(df)

    statistics = calculate_statistics(df)

    # diplay results

    for column, datatype in attribute_types.items():
        print(f"{column} : {datatype}")

    for column, method in encoding.items():
        print(f"{column} : {method}")

    for column, values in ranges.items():
        print(f"{column} : Min = {values[0]}, Max = {values[1]}")

    print("\n===== Missing Values =====")

    for column, count in missing_values.items():
        print(f"{column} : {count}")

    for column, count in outliers.items():
        print(f"{column} : {count}")

    for column, values in statistics.items():

        print(f"\n{column}")

        print(f"Mean               : {values['Mean']}")
        print(f"Variance           : {values['Variance']}")
        print(f"Standard Deviation : {values['Standard Deviation']}")


if __name__ == "__main__":
    main()