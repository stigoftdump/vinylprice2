import pickle
import numpy as np
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# Mapping of quality to numeric value
quality_to_number = {
    'Mint (M) ': 9,
    'Near Mint (NM or M-) ': 8,
    'Very Good Plus (VG+) ': 7,
    'Very Good (VG) ': 6,
    'Good Plus (G+) ': 5,
    'Good (G) ': 4,
    'Fair (F) ': 3,
    'Poor (P) ': 2,
    'Generic ': 1,
    'Not Graded ': 1,
    'No Cover ': 1
}

# list of real prices
real_prices =[
    1.99,
    2.99,
    3.99,
    4.99,
    5.99,
    6.99,
    7.99,
    8.99,
    9.99,
    12.99,
    14.99,
    17.99,
    19.99,
    22.99,
    24.99,
    27.99,
    29.99,
    34.99,
    39.99
]

# function to save processed grid to a file
def save_processed_grid(processed_grid, filename='processed_grid.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(processed_grid, f)

# function to load a saved processed grid from a file
def load_processed_grid(filename='processed_grid.pkl'):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []  # Return an empty list if the file doesn't exist

# converts the record quality and sleeve quality into a score'
def calculate_score(record_quality, sleeve_quality):
    record_value = quality_to_number.get(record_quality, 0)
    sleeve_value = quality_to_number.get(sleeve_quality, 0)
    score = record_value - ((record_value - sleeve_value) / 3)
    return score

# gets the sale price from the calculated price
def realprice(pred_price):
    if pred_price <42.48:
        # Finds the closest price using the min function with a custom key
        foundprice= min(real_prices, key=lambda x: abs(x - pred_price))
    elif pred_price<100:
        nearest_divisible_by_5 = round(pred_price / 5) * 5
        foundprice = nearest_divisible_by_5
    else:
        nearest_divisible_by_10 = round(pred_price / 10) * 10
        foundprice = nearest_divisible_by_10
    return foundprice

def sigmoid_plus_exponential(x, a1, b1, c1, a_exp, b_exp, c_exp, d):
    """
    Combines a sigmoid function with an exponential increase, plus a constant.
    - a1, b1, c1: Parameters for the initial sigmoid rise
    - a_exp, b_exp, c_exp: Parameters for the exponential component
    - d: Base level offset
    """
    x = np.asarray(x)
    first_rise = a1 / (1 + np.exp(-b1 * (x - c1)))
    exponential_increase = a_exp * np.exp(b_exp * (x - c_exp))
    return first_rise + exponential_increase + d

# creates the processed grid data from clipboard data
def make_processed_grid(clipboard_content, start_date):

    # status_message is blank for now
    status_message = None
    processed_grid = None

    # Check for the presence of "Order Date" and "Change Currency" in the clipboard content
    if "Order Date" not in clipboard_content or "Change Currency" not in clipboard_content:
        # return status that no data is there
        return None, "No Discogs Data in text box"
    else:
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
                        # Remove '£' and commas, then convert to a float
                        clean_price = price_str[1:].replace(',', '')
                        filtered_grid[i] = filtered_grid[i][:3] + (float(clean_price),)
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

    return processed_grid, status_message

# writes the chart
def plot_chart(qualities, prices, quality_range, predicted_prices, reqscore, predicted_price, upper_bound, actual_price):
    # Plot the scatter points and curve
    plt.scatter(qualities, prices, color='red', label='Items Sold')
    plt.plot(quality_range, predicted_prices, color='blue', linestyle='-', label='Best Fit')

    # Set y-axis limits
    plt.ylim(0, max(prices) * 1.1)

    # Set x-axis limits to the quality range
    plt.xlim(min(min(qualities), reqscore) * 0.97, max(max(qualities), reqscore) * 1.03)

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
                ymin=0, ymax=actual_price / (max(prices) * 1.1))
    plt.axvline(x=float(reqscore), color='purple', linestyle='--',
                ymin=actual_price / (max(prices) * 1.1),
                ymax=upper_bound / (max(prices) * 1.1))

    # Mark specific points
    plt.scatter(float(reqscore), actual_price, color='green', s=100)
    plt.scatter(float(reqscore), predicted_price, color='orange', s=100)
    plt.scatter(float(reqscore), upper_bound, color='purple', s=100)

    # Add title and labels
    plt.title('Predicted Prices')
    plt.xlabel('Quality')
    plt.ylabel('Price (£)')
    plt.legend()

    # Create a reverse mapping from number to quality text
    number_to_quality = {v: k for k, v in quality_to_number.items()}

    # Gets labels size
    min_quality_label = int(min(min(qualities), reqscore) * 0.97)
    max_quality_label = int(max(max(qualities), reqscore) * 1.03)

    # Generate a list of all whole numbers within this range
    all_quality_numbers = range(min_quality_label, max_quality_label + 1)

    # Generate the corresponding text labels for all whole numbers
    all_quality_labels = [number_to_quality.get(q, str(q)) for q in all_quality_numbers]

    # Set the x-axis ticks and labels
    plt.xticks(all_quality_numbers, all_quality_labels, rotation=15, ha='right')
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping

    plt.savefig('static/chart.png')  # Save as PNG

def return_variables(argument):
    # returns the variables from the argument sent in sys
    if len(argument) < 4:
        print("Error: data is missing.")
        sys.exit(1)
    try:
        reqscore = float(argument[1])  # Read the first argument as reqscore
        shop_var = float(argument[2])  # Read the second argument as shop_var
        start_date = argument[3] # read the third argument as start_date
        add_data = argument[4] # gets the add_data flag
        max_price = argument[5] # gets the max_price
        discogs_data = argument[6]

    except ValueError:
        print("Error: Both reqscore and shop_var must be numbers.")
        sys.exit(1)

    return reqscore, shop_var, start_date, add_data, max_price, discogs_data

def graph_logic(reqscore, shop_var, processed_grid):
    # function that returns everything needed to make a chart
    # needs to return:
    # qualities, prices, X_smooth, y_smooth_pred, reqscore, predicted_price, upper_bound, actual_price, upper_bound

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
        1.5,           # b1: steepness of first rise
        3.0,           # c1: midpoint of first rise (e.g., < 6)
        max(y) * 0.05, # a_exp: scaling for exponential (start smaller)
        0.5,           # b_exp: exponential growth rate (moderate guess)
        7.0,           # c_exp: onset point for exponential (e.g., > 6)
        min(y) if y.size > 0 else 0 # d: base price
    ]

    # Add bounds to ensure monotonicity and reasonable parameters
    # --- Adjust Bounds for Sigmoid + Exponential ---
    # Ensure bounds match the order: [a1, b1, c1, a_exp, b_exp, c_exp, d]
    # Constraint ideas: c1 < 6, c_exp > 6, b_exp > 0
    lower_bounds = [
        0,      # a1 >= 0
        0.1,    # b1 >= 0.1 (avoid zero steepness)
        1,      # c1 >= 1 (within quality range)
        0,      # a_exp >= 0
        0.01,   # b_exp > 0 (ensure growth)
        6,      # c_exp >= 6 (start exponential after sigmoid midpoint)
        0       # d >= 0
    ]
    upper_bounds = [
        np.inf, # a1
        5,      # b1 (limit steepness)
        6,      # c1 <= 6 (sigmoid midpoint before threshold)
        np.inf, # a_exp
        5,      # b_exp (limit growth rate)
        9,      # c_exp <= 9 (within quality range)
        np.inf  # d
    ]
    bounds = (lower_bounds, upper_bounds)

    # Fit the double sigmoid model
    params, _ = curve_fit(
        sigmoid_plus_exponential,  # Use the new function
        X.flatten(),
        y,
        p0=initial_guess,
        bounds=bounds,
        maxfev=100000  # Keep high max iterations, might be needed
    )

    # Extract parameters for readability
    #a1, b1, c1, a_exp, b_exp, c_exp, d = params

    def predict_price_exp(quality_value):
        # Ensure it calls the correct function with the fitted params
        return sigmoid_plus_exponential(quality_value, *params)

    # Calculate predictions for all data points
    y_pred = predict_price_exp(X.flatten())

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Predict price for the requested quality score
    predicted_price = predict_price_exp(reqscore)

    #' new definition of upper bound'
    upper_bound = predicted_price+rmse

    # amends the predicated price to allow for the rmse
    adjusted_price = predicted_price + (rmse*shop_var)

    # gets the actual price
    actual_price = realprice(adjusted_price)

    # Create a smooth curve for plotting the fitted function
    X_smooth = np.linspace(min(X.flatten()), max(X.flatten()), 200)
    y_smooth_pred = predict_price_exp(X_smooth)

    return qualities, prices, X_smooth, y_smooth_pred, reqscore, predicted_price, upper_bound, actual_price, upper_bound