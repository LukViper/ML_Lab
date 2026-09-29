# ================================================================
# LAB 6 - GENAI ASSISTED k-NN CLASSIFICATION
# ================================================================
#
# Dataset:
#     Lab Session Data.xlsx
#     Sheet: marketing_campaign
#
# Lab 6 objectives:
#     A1. Repeat Lab 5 experiments using GenAI
#     A2. Generate unit and functional tests
#     A3. Compare:
#         1. Self-written kNN
#         2. Scikit-learn kNN
#         3. GenAI-assisted kNN
#
# Metrics:
#     Accuracy, Precision, Recall, F1-score
#     Average computational time over 10 runs
#
# ================================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ================================================================
# A1 - DATA PREPROCESSING
# ================================================================


# GenAI-assisted code
def label_encode(series, order=None):
    """Encode a categorical Series into integer values."""

    if order is None:
        categories = sorted(series.dropna().unique())
    else:
        categories = order

    encoding = {
        value: index
        for index, value in enumerate(categories)
    }

    return series.map(encoding), encoding


# GenAI-assisted code
def one_hot_encode(dataframe, columns):
    """Apply one-hot encoding to selected categorical columns."""

    result = dataframe.copy()

    for column in columns:
        categories = sorted(result[column].dropna().unique())

        for category in categories:
            result[f"{column}_{category}"] = (
                result[column] == category
            ).astype(int)

        result.drop(columns=column, inplace=True)

    return result


# GenAI-assisted code
def convert_date_to_numeric(series):
    """Convert dates into days from the earliest date."""

    dates = pd.to_datetime(
        series,
        dayfirst=True,
        errors="coerce"
    )

    return (dates - dates.min()).dt.days.astype(float)


# GenAI-assisted code
def impute_data(dataframe):
    """
    Fill missing values.

    Numeric attributes:
        Median when outliers exist.
        Mean otherwise.

    Categorical attributes:
        Mode.
    """

    result = dataframe.copy()

    numeric_columns = result.select_dtypes(
        include=np.number
    ).columns

    categorical_columns = result.select_dtypes(
        exclude=np.number
    ).columns

    for column in numeric_columns:

        if result[column].isna().any():

            q1 = result[column].quantile(0.25)
            q3 = result[column].quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = (
                (result[column] < lower) |
                (result[column] > upper)
            ).any()

            value = (
                result[column].median()
                if outliers
                else result[column].mean()
            )

            result[column] = result[column].fillna(value)

    for column in categorical_columns:

        if result[column].isna().any():
            result[column] = result[column].fillna(
                result[column].mode()[0]
            )

    return result


# GenAI-assisted code
def prepare_data(dataframe):
    """
    Prepare the marketing campaign dataset for kNN.
    """

    data = dataframe.copy()

    # Response is the classification target.
    y = data["Response"].astype(int).to_numpy()

    # ID is an identifier and must not be used as a feature.
    data.drop(
        columns=["Response", "ID"],
        inplace=True
    )

    # Convert registration date.
    data["Dt_Customer"] = convert_date_to_numeric(
        data["Dt_Customer"]
    )

    # Handle missing values.
    data = impute_data(data)

    # Education is treated as an ordinal attribute.
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

    # Marital status is nominal.
    data = one_hot_encode(
        data,
        ["Marital_Status"]
    )

    # Keep numerical attributes only.
    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns.tolist()

    # Remove constant attributes.
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

    # Normalize features to [0, 1].
    scaler = MinMaxScaler()

    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns

    data[numeric_columns] = scaler.fit_transform(
        data[numeric_columns]
    )

    return (
        data.to_numpy(dtype=float),
        y,
        data.columns.tolist(),
        education_mapping
    )


# ================================================================
# A1 - DISTANCE CALCULATION
# ================================================================


# GenAI-assisted code
def minkowski_distance(a, b, p=2):
    """Calculate Minkowski distance."""

    if p < 1:
        raise ValueError("p must be greater than or equal to 1.")

    return float(
        np.sum(np.abs(a - b) ** p) ** (1 / p)
    )


# GenAI-assisted code
def distances_from_point(test_point, training_data, p=2):
    """Calculate distance from one test vector to all training vectors."""

    difference = np.abs(
        training_data - np.asarray(test_point)
    )

    return np.sum(
        difference ** p,
        axis=1
    ) ** (1 / p)


# ================================================================
# A1 - SORTING
# ================================================================


# GenAI-assisted code
def bubble_sort(items):
    """Sort distance-index pairs using Bubble Sort."""

    items = list(items)

    for end in range(len(items) - 1, 0, -1):

        for index in range(end):

            if items[index] > items[index + 1]:
                items[index], items[index + 1] = (
                    items[index + 1],
                    items[index]
                )

    return items


# GenAI-assisted code
def selection_sort(items):
    """Sort distance-index pairs using Selection Sort."""

    items = list(items)

    for i in range(len(items)):

        smallest = i

        for j in range(i + 1, len(items)):

            if items[j] < items[smallest]:
                smallest = j

        items[i], items[smallest] = (
            items[smallest],
            items[i]
        )

    return items


# GenAI-assisted code
def insertion_sort(items):
    """Sort distance-index pairs using Insertion Sort."""

    items = list(items)

    for i in range(1, len(items)):

        current = items[i]
        position = i - 1

        while (
            position >= 0
            and items[position] > current
        ):
            items[position + 1] = items[position]
            position -= 1

        items[position + 1] = current

    return items


# GenAI-assisted code
def sort_items(items, method="insertion"):
    """Select a sorting algorithm using a configuration parameter."""

    methods = {
        "bubble": bubble_sort,
        "selection": selection_sort,
        "insertion": insertion_sort
    }

    if method not in methods:
        raise ValueError(
            "Choose bubble, selection, or insertion."
        )

    return methods[method](items)


# ================================================================
# A1 - NEIGHBOR IDENTIFICATION
# ================================================================


# GenAI-assisted code
def get_neighbors(
    test_point,
    training_data,
    k,
    p=2,
    sorting_method="insertion"
):
    """Find the k nearest neighbors."""

    if k < 1 or k > len(training_data):
        raise ValueError(
            "k must be between 1 and the number of training samples."
        )

    distances = distances_from_point(
        test_point,
        training_data,
        p
    )

    candidates = [
        (float(distance), index)
        for index, distance in enumerate(distances)
    ]

    # The original index acts as a deterministic
    # tie-breaking mechanism.
    candidates = sort_items(
        candidates,
        sorting_method
    )

    return candidates[:k]


# ================================================================
# A1 - CLASS ASSIGNMENT
# ================================================================


# GenAI-assisted code
def majority_vote(neighbors, labels):
    """Assign a class using normal majority voting."""

    votes = {}

    for distance, index in neighbors:

        label = int(labels[index])
        votes[label] = votes.get(label, 0) + 1

    maximum = max(votes.values())

    candidates = [
        label
        for label, count in votes.items()
        if count == maximum
    ]

    # Smaller class label resolves a voting tie.
    return min(candidates)


# GenAI-assisted code
def weighted_vote(neighbors, labels):
    """Assign a class using inverse-distance weighted voting."""

    scores = {}

    nearest_distance = {}

    for distance, index in neighbors:

        label = int(labels[index])

        nearest_distance[label] = min(
            nearest_distance.get(label, float("inf")),
            distance
        )

        if distance == 0:
            scores[label] = float("inf")
            continue

        if scores.get(label) != float("inf"):
            scores[label] = (
                scores.get(label, 0)
                + 1 / distance
            )

    maximum = max(scores.values())

    candidates = [
        label
        for label, score in scores.items()
        if np.isclose(score, maximum)
    ]

    # First compare the closest neighbor.
    # If still tied, select the smaller label.
    return min(
        candidates,
        key=lambda label: (
            nearest_distance[label],
            label
        )
    )


# ================================================================
# A7 - SELF-WRITTEN kNN IMPLEMENTATION
# ================================================================
#
# This section represents the original implementation from Lab 5.
# It is intentionally kept separate from the GenAI implementation
# so that Lab 6 can compare both implementations.
#
# ================================================================


# Self-written Lab 5 implementation
def self_knn_predict(
    X_train,
    y_train,
    X_test,
    k=3,
    weighted=False
):
    """Predict classes using the Lab 5 custom kNN approach."""

    predictions = []

    for test_point in X_test:

        neighbors = get_neighbors(
            test_point,
            X_train,
            k,
            p=2,
            sorting_method="insertion"
        )

        if weighted:
            prediction = weighted_vote(
                neighbors,
                y_train
            )
        else:
            prediction = majority_vote(
                neighbors,
                y_train
            )

        predictions.append(prediction)

    return np.asarray(predictions)


# ================================================================
# A7 - GenAI kNN IMPLEMENTATION
# ================================================================
#
# GenAI-generated implementation.
#
# This uses a different coding structure from the Lab 5 version.
# ================================================================


# GenAI-assisted implementation
class GenAIKNN:

    def __init__(
        self,
        k=3,
        weighted=False,
        p=2
    ):
        self.k = k
        self.weighted = weighted
        self.p = p
        self.X_train = None
        self.y_train = None

    # GenAI-assisted code
    def fit(self, X, y):
        """Store training samples and their class labels."""

        if self.k < 1 or self.k > len(X):
            raise ValueError(
                "Invalid value of k."
            )

        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)

        return self

    # GenAI-assisted code
    def _nearest(self, sample):

        distances = np.linalg.norm(
            self.X_train - sample,
            ord=self.p,
            axis=1
        )

        order = np.lexsort(
            (
                np.arange(len(distances)),
                distances
            )
        )

        return order[:self.k], distances

    # GenAI-assisted code
    def _classify(self, indexes, distances):

        labels = self.y_train[indexes]

        if not self.weighted:
            values, counts = np.unique(
                labels,
                return_counts=True
            )

            return values[
                np.argmax(counts)
            ]

        weights = {}

        for index in indexes:

            label = self.y_train[index]
            distance = distances[index]

            if distance == 0:
                return label

            weights[label] = (
                weights.get(label, 0)
                + 1 / distance
            )

        return max(
            weights,
            key=weights.get
        )

    # GenAI-assisted code
    def predict(self, X):

        X = np.asarray(X, dtype=float)

        predictions = []

        for sample in X:

            indexes, distances = self._nearest(
                sample
            )

            predictions.append(
                self._classify(
                    indexes,
                    distances
                )
            )

        return np.asarray(predictions)

    # GenAI-assisted code
    def score(self, X, y):

        predictions = self.predict(X)

        return accuracy_score(
            y,
            predictions
        )


# ================================================================
# A2 - UNIT TESTS
# ================================================================


# GenAI-assisted test
def test_distance_function():

    a = np.array([0, 0, 0])
    b = np.array([3, 4, 0])

    distance = minkowski_distance(
        a,
        b,
        p=2
    )

    assert np.isclose(
        distance,
        5.0
    )


# GenAI-assisted test
def test_label_encoding():

    series = pd.Series(
        ["A", "B", "A"]
    )

    encoded, mapping = label_encode(
        series
    )

    assert len(mapping) == 2
    assert encoded.iloc[0] == encoded.iloc[2]


# GenAI-assisted test
def test_sorting_algorithms():

    values = [
        (0.8, 2),
        (0.1, 4),
        (0.4, 1),
        (0.1, 3)
    ]

    expected = sorted(values)

    assert bubble_sort(values) == expected
    assert selection_sort(values) == expected
    assert insertion_sort(values) == expected


# GenAI-assisted test
def test_neighbor_selection():

    X = np.array([
        [0, 0],
        [1, 1],
        [10, 10]
    ])

    neighbors = get_neighbors(
        np.array([0, 0]),
        X,
        k=2
    )

    assert len(neighbors) == 2
    assert neighbors[0][1] == 0


# GenAI-assisted test
def test_majority_vote():

    neighbors = [
        (0.1, 0),
        (0.2, 1),
        (0.3, 2)
    ]

    labels = np.array([
        1,
        1,
        0
    ])

    assert majority_vote(
        neighbors,
        labels
    ) == 1


# GenAI-assisted test
def test_weighted_vote():

    neighbors = [
        (0.1, 1),
        (0.9, 0),
        (1.0, 0)
    ]

    labels = np.array([
        0,
        0,
        1
    ])

    # Class 0 receives the strongest contribution
    # because its nearest neighbor has distance 0.1.
    assert weighted_vote(
        neighbors,
        labels
    ) == 0


# GenAI-assisted functional test
def test_genai_knn():

    X_train = np.array([
        [0, 0],
        [0, 1],
        [10, 10],
        [10, 11]
    ])

    y_train = np.array([
        0,
        0,
        1,
        1
    ])

    X_test = np.array([
        [0, 0.5],
        [10, 10.5]
    ])

    model = GenAIKNN(k=3)

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    assert len(predictions) == 2
    assert predictions[0] == 0
    assert predictions[1] == 1


# ================================================================
# RUN ALL TESTS
# ================================================================


def run_tests():

    tests = [
        ("Distance Function", test_distance_function),
        ("Label Encoding", test_label_encoding),
        ("Sorting Algorithms", test_sorting_algorithms),
        ("Neighbor Selection", test_neighbor_selection),
        ("Majority Voting", test_majority_vote),
        ("Weighted Voting", test_weighted_vote),
        ("GenAI kNN Functional Test", test_genai_knn)
    ]

    print("\n" + "=" * 65)
    print("UNIT AND FUNCTIONAL TEST RESULTS")
    print("=" * 65)

    passed = 0

    for name, test in tests:

        try:
            test()
            print(f"PASS : {name}")
            passed += 1

        except AssertionError:
            print(f"FAIL : {name}")

        except Exception as error:
            print(
                f"FAIL : {name} -> {error}"
            )

    print(
        f"\nTests passed: {passed}/{len(tests)}"
    )

    return passed == len(tests)


# ================================================================
# A3 - PERFORMANCE METRICS
# ================================================================


def calculate_metrics(actual, predicted):

    return {
        "Accuracy": accuracy_score(
            actual,
            predicted
        ),

        "Precision": precision_score(
            actual,
            predicted,
            zero_division=0
        ),

        "Recall": recall_score(
            actual,
            predicted,
            zero_division=0
        ),

        "F1-Score": f1_score(
            actual,
            predicted,
            zero_division=0
        )
    }


# ================================================================
# A3 - 10 RUN PERFORMANCE TEST
# ================================================================

def benchmark_model(model_function, runs=10):
    """
    Measure prediction time over multiple runs.

    The model is prepared before this function is called.
    Therefore, only prediction/inference time is measured.
    """

    execution_times = []
    predictions = None

    for _ in range(runs):

        start_time = time.perf_counter()

        predictions = model_function()

        end_time = time.perf_counter()

        execution_times.append(
            end_time - start_time
        )

    average_time = np.mean(execution_times)

    return predictions, average_time


# ================================================================
# A3 - THREE-WAY COMPARISON
# ================================================================

def compare_models(
    X_train,
    X_test,
    y_train,
    y_test,
    k=3
):
    """
    Compare the three kNN implementations using:
        Accuracy
        Precision
        Recall
        F1-score
        Average prediction time over 10 runs
    """

    # ------------------------------------------------------------
    # 1. Self-written Lab 5 kNN
    # ------------------------------------------------------------

    def run_self_written():

        return self_knn_predict(
            X_train,
            y_train,
            X_test,
            k=k,
            weighted=False
        )

    self_predictions, self_time = benchmark_model(
        run_self_written,
        runs=10
    )

    self_metrics = calculate_metrics(
        y_test,
        self_predictions
    )


    # ------------------------------------------------------------
    # 2. Scikit-learn kNN
    # ------------------------------------------------------------

    sklearn_model = KNeighborsClassifier(
        n_neighbors=k,
        metric="minkowski",
        p=2,
        weights="uniform"
    )

    # Fit only once, outside the timing loop.
    sklearn_model.fit(
        X_train,
        y_train
    )

    def run_sklearn():

        return sklearn_model.predict(
            X_test
        )

    sklearn_predictions, sklearn_time = benchmark_model(
        run_sklearn,
        runs=10
    )

    sklearn_metrics = calculate_metrics(
        y_test,
        sklearn_predictions
    )


    # ------------------------------------------------------------
    # 3. GenAI kNN
    # ------------------------------------------------------------

    genai_model = GenAIKNN(
        k=k,
        weighted=False,
        p=2
    )

    # Fit only once, outside the timing loop.
    genai_model.fit(
        X_train,
        y_train
    )

    def run_genai():

        return genai_model.predict(
            X_test
        )

    genai_predictions, genai_time = benchmark_model(
        run_genai,
        runs=10
    )

    genai_metrics = calculate_metrics(
        y_test,
        genai_predictions
    )


    # ------------------------------------------------------------
    # Create comparison table
    # ------------------------------------------------------------

    results = pd.DataFrame([
        {
            "Implementation": "Self-written kNN",
            **self_metrics,
            "Average Prediction Time (seconds)": self_time
        },
        {
            "Implementation": "Scikit-learn kNN",
            **sklearn_metrics,
            "Average Prediction Time (seconds)": sklearn_time
        },
        {
            "Implementation": "GenAI kNN",
            **genai_metrics,
            "Average Prediction Time (seconds)": genai_time
        }
    ])

    return results


# ================================================================
# A8/A9 - ACCURACY COMPARISON FOR DIFFERENT k VALUES
# ================================================================


def compare_k_values(
    X_train,
    X_test,
    y_train,
    y_test
):

    k_values = [
        1, 3, 5, 7,
        9, 11, 13, 15
    ]

    rows = []

    for k in k_values:

        # Self-written kNN
        self_pred = self_knn_predict(
            X_train,
            y_train,
            X_test,
            k=k
        )

        self_accuracy = accuracy_score(
            y_test,
            self_pred
        )


        # Scikit-learn kNN
        sklearn_model = KNeighborsClassifier(
            n_neighbors=k,
            metric="minkowski",
            p=2,
            weights="uniform"
        )

        sklearn_model.fit(
            X_train,
            y_train
        )

        sklearn_pred = sklearn_model.predict(
            X_test
        )

        sklearn_accuracy = accuracy_score(
            y_test,
            sklearn_pred
        )


        # GenAI kNN
        genai_model = GenAIKNN(
            k=k,
            weighted=False,
            p=2
        )

        genai_model.fit(
            X_train,
            y_train
        )

        genai_pred = genai_model.predict(
            X_test
        )

        genai_accuracy = accuracy_score(
            y_test,
            genai_pred
        )


        rows.append({
            "k": k,
            "Self-written kNN": self_accuracy,
            "Scikit-learn kNN": sklearn_accuracy,
            "GenAI kNN": genai_accuracy
        })

    return pd.DataFrame(rows)


# ================================================================
# MAIN PROGRAM
# ================================================================


def main():

    print("=" * 65)
    print("LAB 6 - GENAI ASSISTED kNN CLASSIFICATION")
    print("=" * 65)


    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------

    file_path = "Lab Session Data.xlsx"

    data = pd.read_excel(
        file_path,
        sheet_name="marketing_campaign"
    )

    print("\nOriginal Dataset Shape:")
    print(data.shape)


    # ------------------------------------------------------------
    # Preprocess project data
    # ------------------------------------------------------------

    X, y, feature_names, education_mapping = prepare_data(
        data
    )

    print("\nProcessed Feature Matrix:")
    print(X.shape)

    print("\nEducation Encoding:")
    print(education_mapping)


    # ------------------------------------------------------------
    # Class distribution
    # ------------------------------------------------------------

    classes, counts = np.unique(
        y,
        return_counts=True
    )

    print("\nClass Distribution:")

    for class_value, count in zip(
        classes,
        counts
    ):
        print(
            f"Class {class_value}: {count}"
        )


    # ------------------------------------------------------------
    # Train-test split
    # ------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    print("\nTrain/Test Split:")
    print(
        "Training samples:",
        len(X_train)
    )
    print(
        "Testing samples :",
        len(X_test)
    )


    # ============================================================
    # A1 - DEMONSTRATION OF SORTING
    # ============================================================

    example = [
        (0.7, 2),
        (0.2, 4),
        (0.2, 1),
        (0.5, 0)
    ]

    print("\nSorting Algorithm Demonstration:")
    print("Original  :", example)
    print(
        "Bubble    :",
        sort_items(example, "bubble")
    )
    print(
        "Selection :",
        sort_items(example, "selection")
    )
    print(
        "Insertion :",
        sort_items(example, "insertion")
    )


    # ============================================================
    # A1 - DISTANCE DEMONSTRATION
    # ============================================================

    a = np.array([1, 2, 3])
    b = np.array([4, 6, 8])

    print("\nDistance Calculation:")

    print(
        "Manhattan Distance:",
        minkowski_distance(a, b, p=1)
    )

    print(
        "Euclidean Distance:",
        minkowski_distance(a, b, p=2)
    )


    # ============================================================
    # A2 - WEIGHTED kNN
    # ============================================================

    weighted_predictions = self_knn_predict(
        X_train,
        y_train,
        X_test,
        k=3,
        weighted=True
    )

    weighted_metrics = calculate_metrics(
        y_test,
        weighted_predictions
    )

    print("\nWeighted kNN Results:")

    for metric, value in weighted_metrics.items():
        print(
            f"{metric}: {value:.4f}"
        )


    # ============================================================
    # A4-A6 - kNN WITH k = 3
    # ============================================================

    print("\n" + "=" * 65)
    print("kNN CLASSIFICATION WITH k = 3")
    print("=" * 65)


    # Self-written implementation
    self_predictions = self_knn_predict(
        X_train,
        y_train,
        X_test,
        k=3
    )

    print("\nSelf-written kNN:")
    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            self_predictions
        )
    )

    print(
        "First 10 predictions:",
        self_predictions[:10]
    )


    # Scikit-learn implementation
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

    sklearn_predictions = sklearn_model.predict(
        X_test
    )

    print("\nScikit-learn kNN:")
    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            sklearn_predictions
        )
    )

    print(
        "First 10 predictions:",
        sklearn_predictions[:10]
    )


    # GenAI implementation
    genai_model = GenAIKNN(
        k=3,
        weighted=False,
        p=2
    )

    genai_model.fit(
        X_train,
        y_train
    )

    genai_predictions = genai_model.predict(
        X_test
    )

    print("\nGenAI kNN:")
    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            genai_predictions
        )
    )

    print(
        "First 10 predictions:",
        genai_predictions[:10]
    )


    # ============================================================
    # A2 - UNIT AND FUNCTIONAL TESTING
    # ============================================================

    run_tests()


    # ============================================================
    # A3 - THREE-WAY PERFORMANCE COMPARISON
    # ============================================================

    print("\n" + "=" * 65)
    print("A3 - THREE-WAY PERFORMANCE COMPARISON")
    print("=" * 65)

    performance_results = compare_models(
        X_train,
        X_test,
        y_train,
        y_test,
        k=3
    )

    print(
        "\nPerformance Comparison "
        "(Average Prediction Time over 10 Runs):"
    )

    print(
        performance_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )


    # ============================================================
    # A8 - COMPARISON FOR DIFFERENT k VALUES
    # ============================================================

    k_results = compare_k_values(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\n" + "=" * 65)
    print("ACCURACY COMPARISON FOR DIFFERENT k VALUES")
    print("=" * 65)

    print(
        k_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


    # ============================================================
    # A8 - ACCURACY PLOT
    # ============================================================

    plt.figure(figsize=(9, 5))

    plt.plot(
        k_results["k"],
        k_results["Self-written kNN"],
        marker="o",
        label="Self-written kNN"
    )

    plt.plot(
        k_results["k"],
        k_results["Scikit-learn kNN"],
        marker="s",
        label="Scikit-learn kNN"
    )

    plt.plot(
        k_results["k"],
        k_results["GenAI kNN"],
        marker="^",
        label="GenAI kNN"
    )

    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Accuracy")

    plt.title(
        "Self-written kNN vs Scikit-learn kNN vs GenAI kNN"
    )

    plt.xticks(
        k_results["k"]
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # ============================================================
    # BEST k VALUES
    # ============================================================

    print("\nBest Results:")

    for column in [
        "Self-written kNN",
        "Scikit-learn kNN",
        "GenAI kNN"
    ]:

        best = k_results.loc[
            k_results[column].idxmax()
        ]

        print(
            f"{column}: "
            f"k = {int(best['k'])}, "
            f"accuracy = {best[column]:.4f}"
        )


# ================================================================
# PROGRAM EXECUTION
# ================================================================


if __name__ == "__main__":
    main()