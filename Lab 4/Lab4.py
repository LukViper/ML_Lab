import pandas as pd
import numpy as np


# ============================================================
# A2 - Label Encoding and One-Hot Encoding
# ============================================================

def label_encode(data, column):
    encoded_data = data.copy()

    categories = pd.factorize(encoded_data[column])[1]

    encoded_data[column] = pd.Categorical(
        encoded_data[column],
        categories=categories
    ).codes

    mapping = {
        category: code
        for code, category in enumerate(categories)
    }

    return encoded_data, mapping


def one_hot_encode(data, column):
    encoded_data = data.copy()

    one_hot_data = pd.get_dummies(
        encoded_data[column],
        prefix=column,
        dtype=int
    )

    encoded_data = pd.concat(
        [
            encoded_data.drop(columns=column),
            one_hot_data
        ],
        axis=1
    )

    return encoded_data


# ============================================================
# A3 - Encoding the marketing_campaign Dataset
# ============================================================

def encode_dataset(data):

    # Education -> Label Encoding
    encoded_data, education_mapping = label_encode(
        data,
        "Education"
    )

    # Marital_Status -> One-Hot Encoding
    encoded_data = one_hot_encode(
        encoded_data,
        "Marital_Status"
    )

    return encoded_data, education_mapping

# ============================================================
# A4 - Minkowski Distance
# ============================================================

def calculate_minkowski_distance(vector_a, vector_b, p):

    differences = [
        abs(a - b) ** p
        for a, b in zip(vector_a, vector_b)
    ]

    return sum(differences) ** (1 / p)


# ============================================================
# A5 - Plot Minkowski Distance for Different Values of p
# ============================================================

import matplotlib.pyplot as plt


def plot_minkowski_distances(vector_a, vector_b):

    p_values = range(1, 11)
    distances = []

    for p in p_values:
        distance = calculate_minkowski_distance(
            vector_a,
            vector_b,
            p
        )

        distances.append(distance)

    plt.figure(figsize=(8, 5))

    plt.plot(
        p_values,
        distances,
        marker="o"
    )

    plt.title("Minkowski Distance vs p")
    plt.xlabel("Value of p")
    plt.ylabel("Minkowski Distance")

    plt.xticks(list(p_values))
    plt.grid(True)

    plt.show()


# ============================================================
# A6 - Compare Minkowski Distance with SciPy
# ============================================================

from scipy.spatial.distance import minkowski


def compare_minkowski_distance(vector_a, vector_b, p):

    custom_result = calculate_minkowski_distance(
        vector_a,
        vector_b,
        p
    )

    scipy_result = minkowski(
        vector_a,
        vector_b,
        p
    )

    return custom_result, scipy_result


# ============================================================
# A7 - Dot Product and Vector Length
# ============================================================

def calculate_dot_product(vector_a, vector_b):

    products = [
        a * b
        for a, b in zip(vector_a, vector_b)
    ]

    return sum(products)


def calculate_vector_length(vector):

    squared_values = [
        value ** 2
        for value in vector
    ]

    return sum(squared_values) ** 0.5

# ============================================================
# A8 - Mean, Variance and Standard Deviation
# ============================================================

def calculate_mean(data):

    return sum(data) / len(data)


def calculate_variance(data):

    mean_value = calculate_mean(data)

    squared_deviations = [
        (value - mean_value) ** 2
        for value in data
    ]

    return sum(squared_deviations) / len(data)


def calculate_standard_deviation(data):

    return calculate_variance(data) ** 0.5


def calculate_dataset_statistics(data):

    numeric_data = data.select_dtypes(include="number")

    means = {}
    variances = {}
    standard_deviations = {}

    for column in numeric_data.columns:

        values = numeric_data[column].dropna()

        means[column] = calculate_mean(values)
        variances[column] = calculate_variance(values)
        standard_deviations[column] = calculate_standard_deviation(values)

    return means, variances, standard_deviations


# ============================================================
# A9 - Compare Statistics with NumPy
# ============================================================

def calculate_numpy_statistics(data):

    numeric_data = data.select_dtypes(include="number")

    numpy_means = numeric_data.mean()
    numpy_standard_deviations = numeric_data.std(ddof=0)

    return numpy_means, numpy_standard_deviations

# ============================================================
# A10 - Histogram and Statistics
# ============================================================

import matplotlib.pyplot as plt


def plot_histogram(data, column):

    values = data[column].dropna()

    plt.figure(figsize=(8, 5))

    plt.hist(
        values,
        bins=10
    )

    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.show()

    mean_value = calculate_mean(values)
    variance_value = calculate_variance(values)

    return mean_value, variance_value


# ============================================================
# A11 - Compare Custom Statistics with Pandas
# ============================================================

def compare_statistics(data, column):

    values = data[column].dropna()

    custom_mean = calculate_mean(values)
    custom_variance = calculate_variance(values)

    pandas_mean = data[column].mean()
    pandas_variance = data[column].var(ddof=0)

    return (
        custom_mean,
        custom_variance,
        pandas_mean,
        pandas_variance
    )



# ============================================================
# Main Program
# ============================================================

def main():

    # Load marketing_campaign sheet from Excel file
    marketing_campaign = pd.read_excel(
        "Lab Session Data.xlsx",
        sheet_name="marketing_campaign"
    )

    # ---------------- A2 ----------------

    print("========== A2 ==========")

    label_encoded_data, education_mapping = label_encode(
        marketing_campaign,
        "Education"
    )

    print("\nEducation Mapping:")
    print(education_mapping)

    print("\nLabel Encoded Education:")
    print(label_encoded_data[["Education"]].head())

    one_hot_data = one_hot_encode(
        marketing_campaign,
        "Marital_Status"
    )

    print("\nOne-Hot Encoded Marital_Status:")
    print(one_hot_data.head())

    # ---------------- A3 ----------------

    print("\n========== A3 ==========")

    encoded_data, education_mapping = encode_dataset(
        marketing_campaign
    )

    print("\nEncoded Dataset:")
    print(encoded_data.head())

    print("\nEducation Mapping:")
    print(education_mapping)

    print("\nFeature Dimensionality:")
    print("Original shape :", marketing_campaign.shape)
    print("Encoded shape  :", encoded_data.shape)

    print(
        "\nChange in number of features:",
        encoded_data.shape[1] - marketing_campaign.shape[1]
    )

    # ---------------- A4 ----------------

    numeric_data = encoded_data.select_dtypes(include="number")

    vector_a = numeric_data.iloc[0].to_numpy()
    vector_b = numeric_data.iloc[1].to_numpy()

    print("\n========== A4 ==========")

    p = int(input("Enter the value of p: "))

    distance = calculate_minkowski_distance(
        vector_a,
        vector_b,
        p
    )

    print(f"\nMinkowski Distance (p = {p}): {distance}")

    # ---------------- A5 ----------------

    print("\n========== A5 ==========")

    plot_minkowski_distances(
        vector_a,
        vector_b
    )

    # ---------------- A6 ----------------

    print("\n========== A6 ==========")

    custom_distance, scipy_distance = compare_minkowski_distance(
        vector_a,
        vector_b,
        p
    )

    print("Custom Minkowski Distance :", custom_distance)
    print("SciPy Minkowski Distance  :", scipy_distance)

    # ---------------- A7 ----------------

    print("\n========== A7 ==========")

    custom_dot_product = calculate_dot_product(
        vector_a,
        vector_b
    )

    numpy_dot_product = np.dot(
        vector_a,
        vector_b
    )

    custom_length_a = calculate_vector_length(vector_a)
    custom_length_b = calculate_vector_length(vector_b)

    numpy_length_a = np.linalg.norm(vector_a)
    numpy_length_b = np.linalg.norm(vector_b)

    print("\nDot Product")
    print("Custom :", custom_dot_product)
    print("NumPy  :", numpy_dot_product)

    print("\nVector A Length")
    print("Custom :", custom_length_a)
    print("NumPy  :", numpy_length_a)

    print("\nVector B Length")
    print("Custom :", custom_length_b)
    print("NumPy  :", numpy_length_b)

    # ---------------- A8 ----------------

    print("\n========== A8 ==========")

    means, variances, standard_deviations = calculate_dataset_statistics(
        encoded_data
    )

    print("\nMeans")
    print(means)

    print("\nVariances")
    print(variances)

    print("\nStandard Deviations")
    print(standard_deviations)

    # ---------------- A9 ----------------

    print("\n========== A9 ==========")

    numpy_means, numpy_standard_deviations = calculate_numpy_statistics(
        encoded_data
    )

    print("\nCustom Mean")
    print(means)

    print("\nNumPy Mean")
    print(numpy_means)

    print("\nCustom Standard Deviation")
    print(standard_deviations)

    print("\nNumPy Standard Deviation")
    print(numpy_standard_deviations)

        # ---------------- A10 ----------------

    print("\n========== A10 ==========")

    income_mean, income_variance = plot_histogram(
        encoded_data,
        "Income"
    )

    print("\nIncome Mean")
    print(income_mean)

    print("\nIncome Variance")
    print(income_variance)


    # ---------------- A11 ----------------

    print("\n========== A11 ==========")

    (
        custom_mean,
        custom_variance,
        pandas_mean,
        pandas_variance
    ) = compare_statistics(
        encoded_data,
        "Income"
    )

    print("\nIncome Statistics Comparison")

    print("\nMean")
    print("Custom :", custom_mean)
    print("Pandas :", pandas_mean)

    print("\nVariance")
    print("Custom :", custom_variance)
    print("Pandas :", pandas_variance)


if __name__ == "__main__":
    main()