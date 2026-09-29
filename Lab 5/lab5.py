import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier


# ============================================================
# A1(a) ENCODING
# ============================================================

def label_encode(series, category_order=None):
    """
    Perform label encoding on a categorical pandas Series.

    If category_order is supplied, it is used to preserve the
    natural order of an ordinal attribute.
    """
    if category_order is None:
        categories = sorted(series.dropna().unique())
    else:
        categories = category_order

    mapping = {
        category: index
        for index, category in enumerate(categories)
    }

    encoded_series = series.map(mapping)

    return encoded_series, mapping


def one_hot_encode(dataframe, columns):
    """
    Perform one-hot encoding on the specified categorical columns.
    """
    encoded_data = dataframe.copy()

    for column in columns:
        categories = sorted(encoded_data[column].dropna().unique())

        for category in categories:
            new_column = f"{column}_{category}"
            encoded_data[new_column] = (
                encoded_data[column] == category
            ).astype(int)

        encoded_data.drop(columns=column, inplace=True)

    return encoded_data


def convert_date_to_numeric(series):
    """
    Convert customer registration dates into the number of days
    since the earliest date in the dataset.
    """
    dates = pd.to_datetime(
        series,
        dayfirst=True,
        errors="coerce"
    )

    return (dates - dates.min()).dt.days.astype(float)


# ============================================================
# A1(b) DATA IMPUTATION
# ============================================================

def impute_data(dataframe):
    """
    Fill missing values using appropriate central tendencies.

    Numeric attributes:
        - Median if outliers are present
        - Mean otherwise

    Categorical attributes:
        - Mode
    """
    result = dataframe.copy()

    numeric_columns = result.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = result.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    # Numeric attributes
    for column in numeric_columns:

        if result[column].isna().any():

            q1 = result[column].quantile(0.25)
            q3 = result[column].quantile(0.75)
            iqr = q3 - q1

            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr

            has_outliers = (
                (result[column] < lower_limit) |
                (result[column] > upper_limit)
            ).any()

            if has_outliers:
                fill_value = result[column].median()
            else:
                fill_value = result[column].mean()

            result[column] = result[column].fillna(fill_value)

    # Categorical attributes
    for column in categorical_columns:

        if result[column].isna().any():
            mode_value = result[column].mode()[0]
            result[column] = result[column].fillna(mode_value)

    return result


# ============================================================
# DATA PREPROCESSING
# ============================================================

def prepare_project_data(dataframe):
    """
    Prepare the marketing_campaign dataset for classification.

    Returns:
        X               -> processed feature matrix
        y               -> target vector
        feature_names   -> names of processed features
        education_map   -> label encoding mapping
    """

    data = dataframe.copy()

    # Target variable
    target = data["Response"].astype(int).to_numpy()

    # Remove target and identifier
    data.drop(
        columns=["Response", "ID"],
        inplace=True
    )

    # Convert date attribute to numeric
    data["Dt_Customer"] = convert_date_to_numeric(
        data["Dt_Customer"]
    )

    # Impute missing values
    data = impute_data(data)

    # Education is ordinal
    education_order = [
        "Basic",
        "2n Cycle",
        "Graduation",
        "Master",
        "PhD"
    ]

    data["Education"], education_mapping = label_encode(
        data["Education"],
        education_order
    )

    # Marital status is nominal
    data = one_hot_encode(
        data,
        ["Marital_Status"]
    )

    # Keep only numerical attributes
    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns.tolist()

    # Remove constant attributes
    constant_columns = [
        column
        for column in numeric_columns
        if data[column].nunique() <= 1
    ]

    if constant_columns:
        data.drop(
            columns=constant_columns,
            inplace=True
        )

        numeric_columns = [
            column
            for column in numeric_columns
            if column not in constant_columns
        ]

    # Normalize numerical features
    scaler = MinMaxScaler()

    data[numeric_columns] = scaler.fit_transform(
        data[numeric_columns]
    )

    X = data.to_numpy(dtype=float)

    feature_names = data.columns.tolist()

    return (
        X,
        target,
        feature_names,
        education_mapping,
        scaler
    )


# ============================================================
# A1(c) DISTANCE CALCULATION
# ============================================================

def minkowski_distance(vector_a, vector_b, p=2):
    """
    Calculate generalized Minkowski distance.

    p = 1 -> Manhattan distance
    p = 2 -> Euclidean distance
    """
    if p < 1:
        raise ValueError(
            "Minkowski order p must be >= 1."
        )

    return float(
        np.sum(
            np.abs(vector_a - vector_b) ** p
        ) ** (1.0 / p)
    )


def calculate_all_distances(test_vector, training_data, p=2):
    """
    Calculate the Minkowski distance between one test vector
    and every vector in the training dataset.
    """
    differences = np.abs(
        training_data - np.asarray(
            test_vector,
            dtype=float
        )
    )

    distances = np.sum(
        differences ** p,
        axis=1
    ) ** (1.0 / p)

    return distances


# ============================================================
# A1(d) THREE SORTING ALGORITHMS
# ============================================================

def bubble_sort(items):
    """
    Bubble Sort implementation.

    Items are tuples:
        (distance, original_index)

    The original index provides deterministic tie breaking.
    """
    result = list(items)
    n = len(result)

    for i in range(n - 1):

        swapped = False

        for j in range(n - 1 - i):

            if result[j] > result[j + 1]:

                result[j], result[j + 1] = (
                    result[j + 1],
                    result[j]
                )

                swapped = True

        if not swapped:
            break

    return result


def selection_sort(items):
    """
    Selection Sort implementation.
    """
    result = list(items)
    n = len(result)

    for i in range(n - 1):

        minimum_index = i

        for j in range(i + 1, n):

            if result[j] < result[minimum_index]:
                minimum_index = j

        if minimum_index != i:

            result[i], result[minimum_index] = (
                result[minimum_index],
                result[i]
            )

    return result


def insertion_sort(items):
    """
    Insertion Sort implementation.
    """
    result = list(items)

    for i in range(1, len(result)):

        key = result[i]
        j = i - 1

        while j >= 0 and result[j] > key:

            result[j + 1] = result[j]
            j -= 1

        result[j + 1] = key

    return result


def sort_neighbors(items, algorithm="insertion"):
    """
    Select the sorting algorithm using a configuration parameter.
    """

    algorithms = {
        "bubble": bubble_sort,
        "selection": selection_sort,
        "insertion": insertion_sort
    }

    if algorithm not in algorithms:
        raise ValueError(
            "algorithm must be 'bubble', "
            "'selection', or 'insertion'."
        )

    return algorithms[algorithm](items)


# ============================================================
# A1(e) IDENTIFY K NEIGHBORS
# ============================================================

def identify_neighbors(
    test_vector,
    training_data,
    k,
    p=2,
    sorting_algorithm="insertion"
):
    """
    Identify the k nearest training vectors.

    Tie breaking for equal distances:
        The vector with the smaller original training index
        is preferred.

    A small sorted neighbour list is maintained instead of
    sorting every training distance, which keeps the custom
    sorting implementation practical for the full dataset.
    """

    if k <= 0 or k > len(training_data):
        raise ValueError(
            "k must be between 1 and the number "
            "of training samples."
        )

    distances = calculate_all_distances(
        test_vector,
        training_data,
        p
    )

    nearest = []

    for index, distance in enumerate(distances):

        candidate = (
            float(distance),
            index
        )

        # Fill the neighbour list initially
        if len(nearest) < k:

            nearest.append(candidate)

            nearest = sort_neighbors(
                nearest,
                sorting_algorithm
            )

        # Replace the current farthest neighbour
        elif candidate < nearest[-1]:

            nearest[-1] = candidate

            nearest = sort_neighbors(
                nearest,
                sorting_algorithm
            )

    return nearest


# ============================================================
# A1(f) CLASS EVALUATION AND ASSIGNMENT
# ============================================================

def assign_class(neighbors, training_labels, weighted=False):
    """
    Assign a class using majority voting.

    For normal kNN:
        Each neighbour contributes one vote.

    For weighted kNN:
        Each neighbour contributes 1 / distance.

    Tie breaking:
        1. Select the class having the nearest neighbour.
        2. If still tied, select the smaller class label.
    """

    if len(neighbors) == 0:
        raise ValueError(
            "At least one neighbour is required."
        )

    class_scores = {}
    nearest_distance = {}

    for distance, index in neighbors:

        label = int(training_labels[index])

        if label not in class_scores:

            class_scores[label] = 0.0
            nearest_distance[label] = distance

        nearest_distance[label] = min(
            nearest_distance[label],
            distance
        )

        # Weighted voting
        if weighted:

            # Exact match gets maximum possible importance
            if distance == 0:
                class_scores[label] = float("inf")

            elif not np.isinf(class_scores[label]):
                class_scores[label] += 1.0 / distance

        # Normal majority voting
        else:
            class_scores[label] += 1.0

    maximum_score = max(
        class_scores.values()
    )

    tied_labels = [
        label
        for label, score in class_scores.items()
        if np.isclose(score, maximum_score)
    ]

    # No tie
    if len(tied_labels) == 1:
        return tied_labels[0]

    # Tie breaking
    tied_labels.sort(
        key=lambda label: (
            nearest_distance[label],
            label
        )
    )

    return tied_labels[0]


# ============================================================
# A7. CUSTOM FIT()
# ============================================================

def fit(
    training_data,
    training_labels,
    k=3,
    p=2,
    sorting_algorithm="insertion",
    weighted=False
):
    """
    Store the training data and configuration for custom kNN.

    kNN is a lazy learning algorithm, so no explicit mathematical
    model is trained during fit().
    """

    if k <= 0 or k > len(training_data):
        raise ValueError(
            "k must be valid for the training dataset."
        )

    model = {
        "training_data": np.asarray(
            training_data,
            dtype=float
        ),
        "training_labels": np.asarray(
            training_labels
        ),
        "k": k,
        "p": p,
        "sorting_algorithm": sorting_algorithm,
        "weighted": weighted
    }

    return model


# ============================================================
# A7. CUSTOM PREDICT()
# ============================================================

def predict(model, test_data):
    """
    Predict class labels for test vectors.
    """

    predictions = []

    for test_vector in np.asarray(
        test_data,
        dtype=float
    ):

        neighbors = identify_neighbors(
            test_vector,
            model["training_data"],
            model["k"],
            model["p"],
            model["sorting_algorithm"]
        )

        predicted_class = assign_class(
            neighbors,
            model["training_labels"],
            model["weighted"]
        )

        predictions.append(predicted_class)

    return np.asarray(predictions)


# ============================================================
# A7. CUSTOM SCORE()
# ============================================================

def score(model, test_data, test_labels):
    """
    Calculate classification accuracy.
    """

    predictions = predict(
        model,
        test_data
    )

    actual_labels = np.asarray(
        test_labels
    )

    return float(
        np.mean(
            predictions == actual_labels
        )
    )


# ============================================================
# A8/A9. EXPERIMENT FOR MULTIPLE K VALUES
# ============================================================

def evaluate_k_values(
    X_train,
    X_test,
    y_train,
    y_test,
    k_values,
    p=2,
    sorting_algorithm="insertion"
):
    """
    Evaluate:
        1. Custom normal kNN
        2. sklearn normal kNN
        3. Custom weighted kNN

    for a range of k values.
    """

    custom_scores = []
    sklearn_scores = []
    weighted_scores = []

    for k in k_values:

        # -----------------------------
        # Custom normal kNN
        # -----------------------------
        custom_model = fit(
            X_train,
            y_train,
            k=k,
            p=p,
            sorting_algorithm=sorting_algorithm,
            weighted=False
        )

        custom_accuracy = score(
            custom_model,
            X_test,
            y_test
        )

        custom_scores.append(
            custom_accuracy
        )

        # -----------------------------
        # sklearn normal kNN
        # -----------------------------
        sklearn_model = KNeighborsClassifier(
            n_neighbors=k,
            metric="minkowski",
            p=p,
            weights="uniform"
        )

        sklearn_model.fit(
            X_train,
            y_train
        )

        sklearn_accuracy = sklearn_model.score(
            X_test,
            y_test
        )

        sklearn_scores.append(
            sklearn_accuracy
        )

        # -----------------------------
        # Custom weighted kNN
        # -----------------------------
        weighted_model = fit(
            X_train,
            y_train,
            k=k,
            p=p,
            sorting_algorithm=sorting_algorithm,
            weighted=True
        )

        weighted_accuracy = score(
            weighted_model,
            X_test,
            y_test
        )

        weighted_scores.append(
            weighted_accuracy
        )

    results = pd.DataFrame({
        "k": k_values,
        "Custom kNN": custom_scores,
        "sklearn kNN": sklearn_scores,
        "Custom Weighted kNN": weighted_scores
    })

    return results


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    # --------------------------------------------------------
    # Load project dataset
    # --------------------------------------------------------

    file_path = "Lab Session Data.xlsx"

    dataframe = pd.read_excel(
        file_path,
        sheet_name="marketing_campaign"
    )

    print("=" * 60)
    print("LAB 5 - kNN CLASSIFICATION")
    print("=" * 60)

    print("\nOriginal dataset shape:")
    print(dataframe.shape)

    # --------------------------------------------------------
    # Prepare project data
    # --------------------------------------------------------

    (
        X,
        y,
        feature_names,
        education_mapping,
        scaler
    ) = prepare_project_data(dataframe)

    print("\nProcessed feature matrix shape:")
    print(X.shape)

    print("\nEducation encoding:")
    print(education_mapping)

    print("\nClass distribution:")
    unique_classes, class_counts = np.unique(
        y,
        return_counts=True
    )

    for class_label, count in zip(
        unique_classes,
        class_counts
    ):
        print(
            f"Class {class_label}: {count}"
        )

    # --------------------------------------------------------
    # A3. Train-test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    print("\nA3 - Train/Test Split")
    print("Training samples:", len(X_train))
    print("Testing samples :", len(X_test))

    # --------------------------------------------------------
    # A1(d). Demonstrate the three sorting algorithms
    # --------------------------------------------------------

    sample_neighbors = [
        (0.7, 2),
        (0.2, 4),
        (0.2, 1),
        (0.5, 0)
    ]

    print("\nA1(d) - Sorting Algorithm Demonstration")

    print(
        "Original:",
        sample_neighbors
    )

    print(
        "Bubble Sort:",
        sort_neighbors(
            sample_neighbors,
            "bubble"
        )
    )

    print(
        "Selection Sort:",
        sort_neighbors(
            sample_neighbors,
            "selection"
        )
    )

    print(
        "Insertion Sort:",
        sort_neighbors(
            sample_neighbors,
            "insertion"
        )
    )

    # --------------------------------------------------------
    # A1(c). Distance calculation demonstration
    # --------------------------------------------------------

    distance_a = np.array([1, 2, 3])
    distance_b = np.array([4, 6, 8])

    print("\nA1(c) - Distance Calculation")

    print(
        "Manhattan distance (p=1):",
        minkowski_distance(
            distance_a,
            distance_b,
            p=1
        )
    )

    print(
        "Euclidean distance (p=2):",
        minkowski_distance(
            distance_a,
            distance_b,
            p=2
        )
    )

    # --------------------------------------------------------
    # A4, A5, A6. sklearn kNN with k=3
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("A4-A6 - kNN WITH k = 3")
    print("=" * 60)

    # A7 - Our implementation
    custom_model = fit(
        X_train,
        y_train,
        k=3,
        p=2,
        sorting_algorithm="insertion",
        weighted=False
    )

    custom_accuracy = score(
        custom_model,
        X_test,
        y_test
    )

    custom_predictions = predict(
        custom_model,
        X_test
    )

    print("\nA7 - Custom kNN")
    print(
        "Accuracy:",
        custom_accuracy
    )

    print(
        "First 10 predictions:",
        custom_predictions[:10]
    )

    # A4-A6 - sklearn
    sklearn_model = KNeighborsClassifier(
        n_neighbors=3,
        metric="minkowski",
        p=2,
        weights="uniform"
    )

    sklearn_model.fit(
        X_train,
        y_train
    )

    sklearn_accuracy = sklearn_model.score(
        X_test,
        y_test
    )

    sklearn_predictions = sklearn_model.predict(
        X_test
    )

    print("\nA4-A6 - sklearn kNN")

    print(
        "Accuracy:",
        sklearn_accuracy
    )

    print(
        "First 10 predictions:",
        sklearn_predictions[:10]
    )

    # --------------------------------------------------------
    # A8 and A9
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("A8-A9 - COMPARATIVE ANALYSIS")
    print("=" * 60)

    k_values = [
        1, 3, 5, 7,
        9, 11, 13, 15
    ]

    results = evaluate_k_values(
        X_train,
        X_test,
        y_train,
        y_test,
        k_values,
        p=2,
        sorting_algorithm="insertion"
    )

    print("\nAccuracy comparison:")
    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    # --------------------------------------------------------
    # Best k values
    # --------------------------------------------------------

    best_custom_row = results.loc[
        results["Custom kNN"].idxmax()
    ]

    best_sklearn_row = results.loc[
        results["sklearn kNN"].idxmax()
    ]

    best_weighted_row = results.loc[
        results["Custom Weighted kNN"].idxmax()
    ]

    print("\nBest Custom kNN:")
    print(
        f"k = {int(best_custom_row['k'])}, "
        f"accuracy = {best_custom_row['Custom kNN']:.4f}"
    )

    print("\nBest sklearn kNN:")
    print(
        f"k = {int(best_sklearn_row['k'])}, "
        f"accuracy = {best_sklearn_row['sklearn kNN']:.4f}"
    )

    print("\nBest Custom Weighted kNN:")
    print(
        f"k = {int(best_weighted_row['k'])}, "
        f"accuracy = {best_weighted_row['Custom Weighted kNN']:.4f}"
    )

    # --------------------------------------------------------
    # A8/A9. Accuracy comparison plot
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        results["k"],
        results["Custom kNN"],
        marker="o",
        label="Custom kNN"
    )

    plt.plot(
        results["k"],
        results["sklearn kNN"],
        marker="s",
        label="sklearn kNN"
    )

    plt.plot(
        results["k"],
        results["Custom Weighted kNN"],
        marker="^",
        label="Custom Weighted kNN"
    )

    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Accuracy")
    plt.title(
        "Custom kNN vs sklearn kNN vs Weighted kNN"
    )

    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# PROGRAM EXECUTION
# ============================================================

if __name__ == "__main__":
    main()