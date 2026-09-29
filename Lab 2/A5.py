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

def extract_first_2_observation(df):
    return df.iloc[:2] # return first 2 obs

def get_binary_cols(df):
    binary_columns = []

    for column in df.columns:
        unique_values = df[column].dropna().unique() 

        if set(unique_values) == {"t","f"}: # binary cols
            binary_columns.append(column)
    return binary_columns

def calc_coefficients(df):
    vec1 = df.iloc[0][get_binary_cols(df)] # binary cols of first obs
    vec2 = df.iloc[1][get_binary_cols(df)] # of second obs
    vec1 = vec1.map({"t":1,"f":0}) # t -> 1, f->0
    vec2 = vec2.map({"t":1,"f":0})
    f11 = ((vec1==1)&(vec2==1)).sum()
    f00 = ((vec1==0)&(vec2==0)).sum()
    f10 = ((vec1==1)&(vec2==0)).sum()
    f01 = ((vec1==0)&(vec2==1)).sum()
    jc = f11 / (f01 + f10 + f11) 
    smc = (f11 + f00) / (f00 + f01 + f10 + f11)
    return jc,smc





def main():
    df = load_dataset("Lab Session Data.xlsx")
    
    df = preprocess_dataset(df)

    first_2_obs = extract_first_2_observation(df)
    print(first_2_obs)
    jc,smc = calc_coefficients(df)
    print(jc)
    print(smc)

if __name__ =="__main__":
    main()