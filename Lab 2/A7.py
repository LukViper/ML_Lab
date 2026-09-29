import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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
        df[column] = pd.to_numeric(df[column]) # convert to numeric

    return df


def prepare_vectors(df):
    df = df.copy()
    df = df.replace({"t": 1, "f": 0}) # t -> 1, f->0

    for column in df.select_dtypes(include="object").columns: # encode remaining categorical cols
        df[column] = pd.factorize(df[column])[0]
    df = df.fillna(0) # missing vals
    return df


def calculate_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2) # dot

    magnitude1 = np.linalg.norm(vec1) # norm
    magnitude2 = np.linalg.norm(vec2) 

    return dot_product / (magnitude1 * magnitude2)

def get_binary_cols(df):
    binary_columns = []

    for column in df.columns:
        unique_values = df[column].dropna().unique() 

        if set(unique_values) == {"t","f"}: # binary cols
            binary_columns.append(column)
    return binary_columns

def calc_coefficients(vec1,vec2):
    f11 = ((vec1==1)&(vec2==1)).sum()
    f00 = ((vec1==0)&(vec2==0)).sum()
    f10 = ((vec1==1)&(vec2==0)).sum()
    f01 = ((vec1==0)&(vec2==1)).sum()
    jc = f11 / (f01 + f10 + f11) 
    smc = (f11 + f00) / (f00 + f01 + f10 + f11)
    return jc,smc

def get_first_20(df):
    return df.iloc[:20]

def plot_heatmap(matrix, title):
    sns.heatmap(matrix, annot=True)
    plt.title(title)
    plt.show()

def main():
    df = load_dataset("Lab Session Data.xlsx")

    df = preprocess_dataset(df)

    first20 = get_first_20(df)

    processed_df = prepare_vectors(first20)

    jc_matrix = np.zeros((20, 20))
    smc_matrix = np.zeros((20, 20))
    cos_matrix = np.zeros((20, 20))

    binary_columns = get_binary_cols(first20)

    for i in range(20):

        for j in range(20):

            binary_vec1 = first20.iloc[i][binary_columns]
            binary_vec2 = first20.iloc[j][binary_columns]

            binary_vec1 = binary_vec1.map({"t": 1, "f": 0}).fillna(0)
            binary_vec2 = binary_vec2.map({"t": 1, "f": 0}).fillna(0)
            jc, smc = calc_coefficients(binary_vec1, binary_vec2)

            jc_matrix[i][j] = jc
            smc_matrix[i][j] = smc

            vec1 = processed_df.iloc[i].to_numpy()
            vec2 = processed_df.iloc[j].to_numpy()

            cos_matrix[i][j] = calculate_cosine_similarity(vec1, vec2)

    print(jc_matrix)
    print(smc_matrix)
    print(cos_matrix)
    plot_heatmap(jc_matrix, "Jaccard Coefficient")

    plot_heatmap(smc_matrix, "Simple Matching Coefficient")

    plot_heatmap(cos_matrix, "Cosine Similarity")



if __name__=="__main__":
    main()