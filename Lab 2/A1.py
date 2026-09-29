import pandas as pd
import numpy as np


def load_dataset(file_path):
    return pd.read_excel(file_path, sheet_name="Purchase data") # built in command for loading dataset


def get_feature_matrix_and_output(df):
    X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].to_numpy() # getting a matrix 
    y = df["Payment (Rs)"].to_numpy()
    return X, y


def calculate_rank(feature_matrix):
    return np.linalg.matrix_rank(feature_matrix) # using built in function for rank


def calculate_product_costs(feature_matrix, output_vector):
    pseudo_inverse = np.linalg.pinv(feature_matrix) # using built in function
    product_costs = pseudo_inverse @ output_vector # matrix multiplication

    return pseudo_inverse, product_costs


def main():
    df = load_dataset("Lab Session Data.xlsx") # loading the dataset

    print(df.head()) # displaying the first five observations

    X, y = get_feature_matrix_and_output(df) # Xc = y

    print("Feature Matrix Shape :", X.shape) # displaying dimensions
    print("Output Vector Shape  :", y.shape)

    print("Rank of Feature Matrix :", calculate_rank(X)) # display rank

    X_pinv, product_costs = calculate_product_costs(X, y) # calculate pseudo inverse and costs of product

    print("Pseudo-Inverse Shape :", X_pinv.shape) # display dimension of pseudo inverse

    print(f"Candy Price : {product_costs[0]}") # display costs
    print(f"Mango Price : {product_costs[1]}")
    print(f"Milk Price  : {product_costs[2]}")


if __name__ == "__main__":
    main()