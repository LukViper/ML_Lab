import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


def load_dataset(file_path):
    return pd.read_excel(file_path, sheet_name="IRCTC Stock Price")  # load dataset


def get_price_data(df):
    return df["Price"].to_numpy()  # extract price column


def calculate_numpy_mean(price_data):
    return np.mean(price_data)  # built-in mean


def calculate_numpy_variance(price_data):
    return np.var(price_data)  # built-in variance


def calculate_custom_mean(price_data):
    total = 0

    for price in price_data:
        total += price

    return total / len(price_data) # return avg 


def calculate_custom_variance(price_data):
    mean = calculate_custom_mean(price_data)

    squared_difference_sum = 0

    for price in price_data:
        squared_difference_sum += (price - mean) ** 2 # difference to the power of 2

    return squared_difference_sum / len(price_data)


def measure_execution_time(function, data):
    execution_times = []

    for _ in range(10):
        start_time = time.perf_counter() # start counter

        function(data) # pass function as a parameter

        end_time = time.perf_counter() # end counter

        execution_times.append(end_time - start_time) # add time elapsed

    return sum(execution_times) / len(execution_times)


def get_wednesday_prices(df):
    return df[df["Day"] == "Wed"]["Price"].to_numpy() 


def get_april_prices(df):
    return df[df["Month"] == "Apr"]["Price"].to_numpy()


def calculate_probability_of_loss(df):
    losses = df["Chg%"].apply(lambda change: change < 0) # lambda function to find negative vals

    return losses.sum() / len(df)


def calculate_probability_of_profit_on_wednesday(df):
    wednesday_data = df[df["Day"] == "Wed"] # get wedesday data

    profits = wednesday_data["Chg%"].apply(lambda change: change > 0) # profits on wednesday

    return profits.sum() / len(wednesday_data)


def calculate_conditional_probability(df):
    wednesday_data = df[df["Day"] == "Wed"] # get wednesday data

    profitable_days = wednesday_data[wednesday_data["Chg%"] > 0] # filter for when chg% is positive

    return len(profitable_days) / len(wednesday_data)


def plot_scatter(df):
    plt.figure(figsize=(8, 5))

    plt.scatter(df["Day"], df["Chg%"]) # x is day and y is chg%

    plt.title("Change Percentage vs Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Chg%")

    plt.grid(True)

    plt.show()


def main():
    df = load_dataset("Lab Session Data.xlsx")

    # get values
    price_data = get_price_data(df) 

    numpy_mean = calculate_numpy_mean(price_data) 
    numpy_variance = calculate_numpy_variance(price_data)

    custom_mean = calculate_custom_mean(price_data)
    custom_variance = calculate_custom_variance(price_data)

    numpy_mean_time = measure_execution_time(calculate_numpy_mean, price_data)
    custom_mean_time = measure_execution_time(calculate_custom_mean, price_data)

    numpy_variance_time = measure_execution_time(calculate_numpy_variance, price_data)
    custom_variance_time = measure_execution_time(calculate_custom_variance, price_data)

    wednesday_mean = calculate_custom_mean(get_wednesday_prices(df))
    april_mean = calculate_custom_mean(get_april_prices(df))

    loss_probability = calculate_probability_of_loss(df)

    profit_wednesday_probability = calculate_probability_of_profit_on_wednesday(df)

    conditional_probability = calculate_conditional_probability(df)

    # display outpupts
    print(f"Population Mean : {numpy_mean}")
    print(f"Custom Mean     : {custom_mean}\n")

    print(f"Population Variance : {numpy_variance}")
    print(f"Custom Variance     : {custom_variance}\n")

    print(f"NumPy Mean Time      : {numpy_mean_time:.10f} seconds")
    print(f"Custom Mean Time     : {custom_mean_time:.10f} seconds\n")

    print(f"NumPy Variance Time  : {numpy_variance_time:.10f} seconds")
    print(f"Custom Variance Time : {custom_variance_time:.10f} seconds\n")

    print(f"Wednesday Mean : {wednesday_mean}") # observation is that wednesday produced less than avg profit
    print(f"April Mean     : {april_mean}\n") # observation is that april produced more than avg profit

    print(f"Probability of Loss                : {loss_probability}")
    print(f"Probability of Profit on Wednesday : {profit_wednesday_probability}")
    print(f"Conditional Probability            : {conditional_probability}")

    plot_scatter(df)


if __name__ == "__main__":
    main()