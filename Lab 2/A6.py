import pandas as pd
import numpy as np


def load_dataset(file_path):
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")  # load dataset


def preprocess_dataset(df):
    df = df.replace("?", np.nan).infer_objects(copy=False)

    numeric_columns = [
        "TSH",
        "T3",
        "TT4",
        "T4U",
        "FTI",
        "TBG"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column])

    return df


def prepare_vectors(df):
    df = df.copy()
    df = df.replace({"t": 1, "f": 0}) # t -> 1, f->0

    for column in df.select_dtypes(include="object").columns: # encode remaining categorical cols
        df[column] = pd.factorize(df[column])[0]
    df = df.fillna(0) # missing vals

    vec1 = df.iloc[0].to_numpy()
    vec2 = df.iloc[1].to_numpy()

    return vec1, vec2


def calculate_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2) # dot

    magnitude1 = np.linalg.norm(vec1) # norm
    magnitude2 = np.linalg.norm(vec2) 

    return dot_product / (magnitude1 * magnitude2)


def main():
    df = load_dataset("Lab Session Data.xlsx")

    df = preprocess_dataset(df)

    vec1, vec2 = prepare_vectors(df)

    cosine_similarity = calculate_cosine_similarity(vec1, vec2)

    print(f"Cosine Similarity : {cosine_similarity}")


if __name__ == "__main__":
    main()