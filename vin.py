import pyperclip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sys
from datetime import datetime
from functions import *
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# at the moment, we only have runtype = 1 for a full run
runtype = 1

def main():
    # only import the args as variables if we need to, we won't need to on recalcs.
    if runtype==1:
        # gets the inputs from the sys arguments
        if len(sys.argv) < 4:
            print("Error: data is missing.")
            sys.exit(1)
        try:
            reqscore = float(sys.argv[1])  # Read the first argument as reqscore
            shop_var = float(sys.argv[2])  # Read the second argument as shop_var
            start_date = sys.argv[3] # read the third argument as start_date
            add_data = sys.argv[4] # gets the add_data flag
            max_price = sys.argv[5] # gets the max_price
        except ValueError:
            print("Error: Both reqscore and shop_var must be numbers.")
            sys.exit(1)

        # Ensure max_price is either an integer or None
        if max_price and max_price.isdigit():  # Check if it's not empty and is a number
            max_price = int(max_price)
        else:
            max_price = None  # Set to None explicitly if it's empty or invalid

    # if its a full run, get the processed_grid from the clipboard.
    # at the end of this part, processed_grid is populated
    if runtype==1:
        # Get clipboard content
        clipboard_content = pyperclip.paste()

        # Check for the presence of "Order Date" and "Change Currency" in the clipboard content
        if "Order Date" not in clipboard_content or "Change Currency" not in clipboard_content:
            # Show error message box
            show_error_message("No Discogs data in clipboard. Go to the Discogs Sales History, CTRL-A to select all, CTRL-C to copy, then come back and re-run")
            sys.exit(0)  # Exit the script

        # Split content into rows based on newlines
        rows = clipboard_content.splitlines()  # 'rows' is defined here

        # Extract the portion of the clipboard content starting from "Order Date" and stopping before "Change Currency"
        start_index = None
        end_index = None

        for i, row in enumerate(rows):
            if "Order Date" in row:
                start_index = i
            if "Change Currency" in row and start_index is not None:
                end_index = i
                break

        # Ensure valid indices are found
        if start_index is not None and end_index is not None:
            rows = rows[start_index:end_index]

        # Split each row into cells based on tabs ('\t') and convert to tuples
        grid = [tuple(row.split('\t')) for row in rows]

        # Exclude header row by skipping the first row (assuming it's the header)
        # also removes any purchases from before the start date
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
        filtered_grid = [
            row for row in grid[1:]
            if row[0].strip() and datetime.strptime(row[0].strip(), '%Y-%m-%d').date() >= start_date_obj
            ]

        # Convert the fourth element (index 3) from a string with '£' to a number
        for i in range(len(filtered_grid)):
            if len(filtered_grid[i]) > 3:  # Ensure there is a fourth element
                price_str = filtered_grid[i][3]
                if price_str.startswith('£'):  # Check if the string starts with '£'
                    try:
                        # Remove '£' and convert to a float
                        filtered_grid[i] = filtered_grid[i][:3] + (float(price_str[1:]),)
                    except ValueError:
                        print(f"Error converting {price_str} to a number.")

        # Process each row to calculate the score for the record and sleeve qualities
        processed_grid = []
        for row in filtered_grid:
            if len(row) > 2:  # Ensure there are enough elements (record and sleeve qualities)
                record_quality = row[1]  # Second element (record quality)
                sleeve_quality = row[2]  # Third element (sleeve quality)
                score = calculate_score(record_quality, sleeve_quality)
                # Add the score as the last element in the tuple
                processed_grid.append(row + (score,))
            else:
                # In case there are rows with missing data
                processed_grid.append(row + (None,))

        # adds in the maxprice
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
    params, _ = curve_fit(double_sigmoid, X.flatten(), y, p0=initial_guess, bounds=bounds, maxfev=10000)

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

    # Plot the scatter points and curve
    plt.scatter(qualities, prices, color='red', label='Items Sold')
    plt.plot(quality_range, predicted_prices, color='blue', linestyle='-', label='Best Fit')

    # Set y-axis limits
    plt.ylim(0, max(prices) * 1.1)

    # Set x-axis limits to the quality range
    plt.xlim(1, 9)

    # Add grid for easier reading
    plt.grid(True, linestyle='--', alpha=0.7)

    # Horizontal reference lines
    plt.axhline(y=predicted_price, color='orange', linestyle='--',
                xmin=0, xmax=(float(reqscore) - 1) / 8,  # Adjusted for 1-9 range
                label='Predicted Price')
    plt.axhline(y=upper_bound, color='purple', linestyle='--',
                xmin=0, xmax=(float(reqscore) - 1) / 8,  # Adjusted for 1-9 range
                label='Upper Bound (RMSE)')
    plt.axhline(y=actual_price, color='green', linestyle='--',
                xmin=0, xmax=(float(reqscore) - 1) / 8,  # Adjusted for 1-9 range
                label='Actual Price')

    # Vertical reference line
    plt.axvline(x=float(reqscore), color='green', linestyle='--',
                ymin=0, ymax=(actual_price) / (max(prices) * 1.1))
    plt.axvline(x=float(reqscore), color='purple', linestyle='--',
                ymin=(actual_price) / (max(prices) * 1.1),
                ymax=(upper_bound) / (max(prices) * 1.1))

    # Mark specific points
    plt.scatter(float(reqscore), actual_price, color='green', s=100)
    plt.scatter(float(reqscore), predicted_price, color='orange', s=100)
    plt.scatter(float(reqscore), upper_bound, color='purple', s=100)

    # Add title and labels
    plt.title('Predicted Prices')
    plt.xlabel('Quality')
    plt.ylabel('Price (£)')
    plt.legend()

    # Annotate the turning point
    #plt.axvline(x=6, color='gray', linestyle=':', alpha=0.5)
    #plt.text(6.1, min(prices), 'Turning Point (6)', fontsize=9, color='gray')
    plt.savefig('static/images/chart.png')  # Save as PNG

    # output for sending to flask
    print(f"{round(predicted_price,2)},{round(upper_bound,2)},{round(actual_price,2)}")

if __name__ == "__main__":
    main()