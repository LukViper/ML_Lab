# A1 
# Ordinal - Education, Response
# Nominal - Marital_Status,  ID , Accepted*, Complain, Nominal
# Ratio - Recency, Mnt*, Num* , Z_CostContact, Z_Revenue,  Income, Kidhome, Teenhome,
# IntervaL - Dt_Customer, Year_Birth

# A2 
import pandas as pd
import numpy as np

def label_encode(df, column):
    encoded_df = df.copy()

    unique_values = encoded_df[column].unique() # get unique categories

    mapping = {}

    for index, value in enumerate(unique_values):
        mapping[value] = index # map categorical value to its index 

    encoded_df[column] = encoded_df[column].map(mapping) 

    return encoded_df, mapping


def one_hot_encode(df, column):
    encoded_df = df.copy()

    unique_values = encoded_df[column].unique()  # get unique categories

    for value in unique_values:
        encoded_df[f"{column}_{value}"] = (encoded_df[column] == value).astype(int)  # create binary col

    encoded_df = encoded_df.drop(columns=[column])  # remove original col

    return encoded_df

# A3
def encode_dataset(df):

    encoded_df, education_mapping = label_encode(df, "Education")

    encoded_df = one_hot_encode(encoded_df, "Marital_Status")

    return encoded_df, education_mapping

#A4
def calculate_minkowski_distance(vec1, vec2, p):

    distance = 0

    for value1, value2 in zip(vec1, vec2):
        distance += abs(value1 - value2) ** p  # sum of powered absolute differences

    return distance ** (1 / p)  # take pth root

# A5
import matplotlib.pyplot as plt
def plot_minkowski_distances(vec1, vec2):

    p_values = []
    distances = []

    for p in range(1, 11):
        distance = calculate_minkowski_distance(vec1, vec2, p)  # calculate distance for current p

        p_values.append(p)  # store p value
        distances.append(distance)  # store corresponding distance

    plt.plot(p_values, distances, marker="o")  # plot distance against p

    plt.title("Minkowski Distance vs p")
    plt.xlabel("p")
    plt.ylabel("Distance")

    plt.xticks(range(1, 11))

    plt.grid(True)

    plt.show()

# A6
from scipy.spatial.distance import minkowski
def compare_minkowski_distance(vec1, vec2, p):

    custom_distance = calculate_minkowski_distance(vec1, vec2, p)  # calculate using own function

    scipy_distance = minkowski(vec1, vec2, p)  # calculate using scipy

    return custom_distance, scipy_distance

# A7
def calculate_dot_product(vec1, vec2):

    dot_product = 0

    for value1, value2 in zip(vec1, vec2):
        dot_product += value1 * value2  # multiply corresponding elements

    return dot_product


def calculate_vector_length(vec):
    

    length = 0

    for value in vec:
        length += value ** 2  # sum of squares

    return length ** 0.5  # square root of the sum


# A8
def calculate_mean(data):

    return sum(data) / len(data)


def calculate_variance(data):

    mean = calculate_mean(data)

    variance = 0

    for value in data:
        variance += (value - mean) ** 2  # sum squared differences from mean

    return variance / len(data)


def calculate_standard_deviation(data):

    return calculate_variance(data) ** 0.5  # square root of variance


def calculate_dataset_statistics(df):

    numeric_df = df.select_dtypes(include=["number"])  # select numeric columns

    means = {}
    variances = {}
    standard_deviations = {}

    for column in numeric_df.columns:

        data = numeric_df[column]

        means[column] = calculate_mean(data)  # calculate mean

        variances[column] = calculate_variance(data)  # calculate variance

        standard_deviations[column] = calculate_standard_deviation(data)  # calculate standard deviation

    return means, variances, standard_deviations

# A9
def calculate_numpy_statistics(df):

    numeric_df = df.select_dtypes(include=["number"])  # select numeric columns

    mean = numeric_df.mean()  # calculate mean using numpy

    standard_deviation = numeric_df.std()  # calculate standard deviation using numpy

    return mean, standard_deviation

# A10
def plot_histogram(df, column):

    data = df[column].dropna()  # remove missing values

    plt.hist(data, bins=10)  # plot histogram

    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.grid(True)

    plt.show()

    mean = calculate_mean(data)  # calculate mean

    variance = calculate_variance(data)  # calculate variance

    return mean, variance

def main():
    df = pd.read_excel("Lab Session Data.xlsx",sheet_name="marketing_campaign")
    # A2
    encoded_df, education_mapping = label_encode(df, "Education")
    print("Education Mapping")
    print(education_mapping)
    print(encoded_df[["Education"]].head())
    encoded_df = one_hot_encode(df, "Marital_Status")
    print(encoded_df.head())

    # A3
    encoded_df, education_mapping = encode_dataset(df)
    print("\nEncoded Dataset\n")
    print(encoded_df.head())
    print("\nEducation Mapping")
    print(education_mapping)
    print("\nOriginal Shape :", df.shape)
    print("Encoded Shape  :", encoded_df.shape)

    # A4
    numeric_df = encoded_df.select_dtypes(include=["number"])
    vec1 = numeric_df.iloc[0]
    vec2 = numeric_df.iloc[1]
    p = int(input("Enter value of p: "))
    distance = calculate_minkowski_distance(vec1, vec2, p)
    print(f"\nMinkowski Distance (p = {p}) : {distance}")

    # A5
    plot_minkowski_distances(vec1, vec2)

    # A6
    custom_distance, scipy_distance = compare_minkowski_distance(vec1, vec2, p)
    print(f"\nCustom Minkowski Distance : {custom_distance}")
    print(f"SciPy Minkowski Distance  : {scipy_distance}")

    # A7
    custom_dot = calculate_dot_product(vec1, vec2)
    numpy_dot = np.dot(vec1, vec2)
    custom_length1 = calculate_vector_length(vec1)
    custom_length2 = calculate_vector_length(vec2)
    numpy_length1 = np.linalg.norm(vec1)
    numpy_length2 = np.linalg.norm(vec2)

    print("\nDot Product")
    print("Custom :", custom_dot)
    print("NumPy  :", numpy_dot)
    print("\nVector 1 Length")
    print("Custom :", custom_length1)
    print("NumPy  :", numpy_length1)
    print("\nVector 2 Length")
    print("Custom :", custom_length2)
    print("NumPy  :", numpy_length2)


    # A8
    means, variances, standard_deviations = calculate_dataset_statistics(encoded_df)
    print("\nMeans")
    print(means)
    print("\nVariances")
    print(variances)
    print("\nStandard Deviations")
    print(standard_deviations)

    # A9
    numpy_mean, numpy_standard_deviation = calculate_numpy_statistics(encoded_df)
    print("\nCustom Mean")
    print(means)
    print("\nNumPy Mean")
    print(numpy_mean)
    print("\nCustom Standard Deviation")
    print(standard_deviations)
    print("\nNumPy Standard Deviation")
    print(numpy_standard_deviation)

    # A10
    mean, variance = plot_histogram(encoded_df, "Income")
    print("\nIncome Mean")
    print(mean)
    print("\nIncome Variance")
    print(variance)

if __name__=="__main__":
    main()