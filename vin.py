import pyperclip
import sys
from functions import *
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

def main():
    # gets the inputs from the sys arguments
    reqscore, shop_var, start_date, add_data, max_price = return_variables(sys.argv)

    # Ensure max_price is either an integer or None
    if max_price and max_price.isdigit():  # Check if it's not empty and is a number
        max_price = int(max_price)
    else:
        max_price = None  # Set to None explicitly if it's empty or invalid

    # Gets the processed_grid from the clipboard
    processed_grid = make_processed_grid(pyperclip.paste(), start_date)

    # adds in the maxprice and deletes if anything is above it.
    if max_price is not None:
        processed_grid = [
            row for row in processed_grid
            if len(row) > 3 and isinstance(row[3], (int, float))  and row[3] < max_price
        ]

    # If add_data is True, load the previously saved processed_grid and add it to the current one
    if add_data == "True":
        saved_processed_grid = load_processed_grid()
        processed_grid.extend(saved_processed_grid)

    # Save processed grid to file
    save_processed_grid(processed_grid)

    qualities = []
    prices = []

    for row in processed_grid:
        if len(row) >= 5:  # Ensure there are at least 5 elements
            qualities.append(row[4])  # Quality column (score)
            prices.append(row[3])  # Price column

    # Convert to numpy arrays for regression
    X = np.array(qualities).reshape(-1, 1)  # Quality as independent variable
    y = np.array(prices)  # Price as dependent variable

    # Initial parameter guesses - adjusted for quality range 1-9
    # with turning point at quality 6
    initial_guess = [
        max(y) * 0.4,  # a1: first rise contribution (40% of price range)
        1.5,  # b1: steepness of first rise
        3.0,  # c1: midpoint of first rise (before the turning point at 6)
        max(y) * 0.5,  # a2: second rise contribution (50% of price range)
        0.8,  # b2: steepness of second rise
        7.5,  # c2: midpoint of second rise (after the turning point at 6)
        min(y)  # d: base price
    ]

    # Add bounds to ensure monotonicity and reasonable parameters
    bounds = (
        [0, 0.1, 1, 0, 0.1, 6, 0],  # Lower bounds - c1 ≥ 1, c2 ≥ 6
        [np.inf, 5, 6, np.inf, 5, 9, np.inf]  # Upper bounds - c1 ≤ 6, c2 ≤ 9
    )

    # Fit the double sigmoid model
    params, _ = curve_fit(double_sigmoid, X.flatten(), y, p0=initial_guess, bounds=bounds, maxfev=100000)

    # Extract parameters for readability
    a1, b1, c1, a2, b2, c2, d = params

    def predict_price(quality_value):
        return double_sigmoid(quality_value, *params)

    # Calculate predictions for all data points
    y_pred = predict_price(X.flatten())

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Predict price for the requested quality score
    predicted_price = predict_price(reqscore)

    #' new definition of upper bound'
    upper_bound = predicted_price+rmse

    # amends the predicated price to allow for the rmse
    adjusted_price = predicted_price + (rmse*shop_var)

    # gets the actual price
    actual_price = realprice(adjusted_price)

    # Generate smooth data points for plotting the sigmoid curve
    quality_range = np.linspace(1, 9, 100)  # Quality range from 1 to 9
    predicted_prices = predict_price(quality_range)

    # calls the function to plot the chart and saves it
    plot_chart(qualities, prices, quality_range, predicted_prices, reqscore, predicted_price, upper_bound, actual_price)

    # output for sending to flask
    print(f"{round(predicted_price,2)},{round(upper_bound,2)},{round(actual_price,2)}")

if __name__ == "__main__":
    main()