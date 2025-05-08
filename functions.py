import numpy as np
from scipy.optimize import curve_fit
from grid_functions import realprice
import pickle

UNIFIED_SAVE_FILE = "vinylpricedata.pkl"

def read_application_data():
    """Reads the entire data dictionary from the unified pickle file."""
    try:
        with open(UNIFIED_SAVE_FILE, 'rb') as f:
            data = pickle.load(f)
            return data
    except (FileNotFoundError, EOFError, pickle.UnpicklingError): # Handle empty or corrupted file
        return {} # Return an empty dict if file not found or unpickling error

def write_application_data(data):
    """Writes the entire data dictionary to the unified pickle file."""
    try:
        with open(UNIFIED_SAVE_FILE, 'wb') as f:
            pickle.dump(data, f)
    except IOError as e:
        print(f"Error writing to {UNIFIED_SAVE_FILE}: {e}")

def read_save_value():
    """
    Reads the 'shop_var' value from the unified save file.
    Defaults to 0.8 if not found.
    """
    data = read_application_data()
    shop_var = data.get("shop_var", 0.8)

    return shop_var

def write_save_value(value):
    """
    Writes the given 'shop_var' value to the unified save file.
    Args:
        value (float): The 'shop_var' value to save.
    """
    data = read_application_data() # Read existing data
    data["shop_var"] = value
    write_application_data(data)

def load_processed_grid(filename=UNIFIED_SAVE_FILE):
    """
    Loads the processed grid data from the unified pickle file.
    """
    data = read_application_data()
    grid_data = data.get("processed_grid", [])

    return grid_data

def save_processed_grid(processed_grid, filename=UNIFIED_SAVE_FILE):
    """
    Saves the processed grid data to the unified pickle file.
    """
    data = read_application_data()
    data["processed_grid"] = processed_grid
    write_application_data(data)

def sigmoid_plus_exponential(x, a1, b1, c1, a_exp, b_exp, c_exp, d):
    """
    Calculates the value of a combined sigmoid and exponential function.

    This function models a curve that initially rises sigmoidally and then
    transitions to exponential growth.

    Args:
        x (array-like): The input value(s) (e.g., quality score).
        a1 (float): Amplitude of the sigmoid component.
        b1 (float): Steepness of the sigmoid component.
        c1 (float): Midpoint (horizontal shift) of the sigmoid component.
        a_exp (float): Amplitude scaling factor for the exponential component.
        b_exp (float): Growth rate of the exponential component.
        c_exp (float): Onset point (horizontal shift) for the exponential component.
        d (float): Vertical shift (base value) of the combined function.

    Returns:
        np.ndarray: The calculated value(s) of the combined function.
    """
    x = np.asarray(x)
    first_rise = a1 / (1 + np.exp(-b1 * (x - c1)))
    exponential_increase = a_exp * np.exp(b_exp * (x - c_exp))
    return first_rise + exponential_increase + d

def graph_logic(reqscore, shop_var, processed_grid):
    """
    Performs curve fitting and price prediction based on processed sales data.

    Fits a sigmoid-plus-exponential model to the quality vs. price data,
    calculates predicted prices, error bounds, and generates data for plotting.

    Args:
        reqscore (float): The target quality score for which to predict the price.
        shop_var (float): A factor applied to the calculated upper price bound.
        processed_grid (list): A list of tuples, where each tuple represents a processed
                               sale record (containing price, quality score, etc.).

    Returns:
        tuple: A tuple containing:
            - qualities (list): Original quality scores from the data.
            - prices (list): Original (inflation-adjusted) prices from the data.
            - X_smooth (np.ndarray): Quality values for plotting the smooth fitted curve.
            - y_smooth_pred (np.ndarray): Predicted prices corresponding to X_smooth.
            - predicted_price (float): The predicted price for the reqscore based on the fitted curve.
            - upper_bound (float): The calculated upper bound price based on percentiles.
            - actual_price (float): The final price after applying shop_var and rounding.
    """
    # function that returns everything needed to make a chart

    qualities = []
    prices = []

    for row in processed_grid:
        if len(row) >= 5:  # Ensure there are at least 5 elements
            qualities.append(row[5])  # Quality column (score)
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

    # Calculate predictions for all data points
    def predict_price_exp(quality_value):
        """Predicts price using the fitted sigmoid_plus_exponential model."""
        # Ensure it calls the correct function with the fitted params
        return sigmoid_plus_exponential(quality_value, *params)

    y_pred = predict_price_exp(X.flatten())

    # gets percentile price above line
    # NOTE: The shop_var is now applied to this percentile price
    percentile_price_above_line, percentile_message = get_percentile_price_above_line(y, y_pred, X, reqscore, percentile=90) # Using 90th percentile

    # calculates prices
    predicted_price = predict_price_exp(reqscore)
    upper_bound = percentile_price_above_line # Upper bound is now the calculated percentile price
    adjusted_price = predicted_price + ((upper_bound-predicted_price) * shop_var) # Apply shop_var to the upper bound
    actual_price = realprice(float(adjusted_price))

    # Create a smooth curve for plotting the fitted function
    # Determine the minimum x-value for the smooth curve
    min_x_for_smooth = min(min(X.flatten()), reqscore) if X.size > 0 else reqscore

    # Determine the maximum x-value for the smooth curve
    max_x_for_smooth = max(max(X.flatten()), reqscore) if X.size > 0 else reqscore

    # Create a smooth curve for plotting the fitted function, extending to reqscore if necessary
    X_smooth = np.linspace(min_x_for_smooth, max_x_for_smooth, 200)

    y_smooth_pred = predict_price_exp(X_smooth)

    return qualities, prices, X_smooth, y_smooth_pred, predicted_price, upper_bound, actual_price, percentile_message


def get_percentile_price_above_line(y_true, y_pred, X_quality, reqscore, percentile=90):
    """
    Calculates a localized percentile of the actual prices for data points
    where the actual price is above the predicted price line.

    It attempts to find the specified percentile of these actual prices for points
    close to the requested quality score (reqscore) in increasing bin sizes.
    If insufficient points are found locally, it falls back to calculating
    the percentile from all points above the line.

    Args:
        y_true (np.ndarray): The actual price values.
        y_pred (np.ndarray): The predicted price values from the model.
        X_quality (np.ndarray): The quality scores corresponding to y_true/y_pred.
                                Expected to be a 2D array from reshape(-1,1).
        reqscore (float): The target quality score around which to localize the percentile.
        percentile (float): The desired percentile to calculate (0 to 100).

    Returns:
        float: The calculated local percentile of actual prices for points above
               the best fit line, or the predicted price at reqscore if no points
               are above the line or insufficient points are found. Returns 0.0 as a fallback
               if no points are above the line.
    """
    all_residuals = y_true - y_pred

    # Identify points above the line of best fit (positive residuals)
    points_above_line_indices = np.where(all_residuals > 0)[0]

    # Get the actual prices for points above the line
    y_true_above = y_true[points_above_line_indices]
    X_quality_flat_above = X_quality.flatten()[points_above_line_indices]

    # If no points are above the line, return 0.0 as a fallback
    if len(y_true_above) == 0:
        print("Warning: No sales points found above the line of best fit. Returning 0.0 as upper bound.")
        return 0.0


    # Fallback percentile: percentile of *all* actual prices from points above the line
    fallback_percentile_price = np.percentile(y_true_above, percentile) if len(y_true_above) > 0 else 0.0

    local_percentile_price = fallback_percentile_price  # Default to the fallback
    min_points_in_bin = 3  # Minimum points needed in a bin to calculate local percentile

    # Tier 1: Exact reqscore (using X_quality_flat_above and y_true_above)
    indices_in_bin_tier1 = np.where(np.isclose(X_quality_flat_above, reqscore))[0]
    y_true_in_bin_tier1 = y_true_above[indices_in_bin_tier1]

    if len(y_true_in_bin_tier1) >= min_points_in_bin:
        local_percentile_price = np.percentile(y_true_in_bin_tier1, percentile)
        print(
            f"Using local {percentile}th percentile ({local_percentile_price:.2f}) for points above line at exact quality score {reqscore} based on {len(y_true_in_bin_tier1)} points.")
        percentile_message="Max Price calculated using exact quality"
    else:
        # Tier 2: 0.5 either side of reqscore
        lower_bound_tier2 = reqscore - 0.5
        upper_bound_tier2 = reqscore + 0.5
        indices_in_bin_tier2 = np.where((X_quality_flat_above >= lower_bound_tier2) & (X_quality_flat_above <= upper_bound_tier2))[0]
        y_true_in_bin_tier2 = y_true_above[indices_in_bin_tier2]


        if len(y_true_in_bin_tier2) >= min_points_in_bin:
            local_percentile_price = np.percentile(y_true_in_bin_tier2, percentile)
            print(
                f"Using local {percentile}th percentile ({local_percentile_price:.2f}) for points above line in quality range [{lower_bound_tier2:.2f}, {upper_bound_tier2:.2f}] based on {len(y_true_in_bin_tier2)} points.")
            percentile_message = "Max Price calculated using narrow band quality"
        else:
            # Tier 3: 1.0 either side of reqscore
            lower_bound_tier3 = reqscore - 1.0
            upper_bound_tier3 = reqscore + 1.0
            indices_in_bin_tier3 = np.where((X_quality_flat_above >= lower_bound_tier3) & (X_quality_flat_above <= upper_bound_tier3))[0]
            y_true_in_bin_tier3 = y_true_above[indices_in_bin_tier3]

            if len(y_true_in_bin_tier3) >= min_points_in_bin:
                local_percentile_price = np.percentile(y_true_in_bin_tier3, percentile)
                print(
                    f"Using local {percentile}th percentile ({local_percentile_price:.2f}) for points above line in quality range [{lower_bound_tier3:.2f}, {upper_bound_tier3:.2f}] based on {len(y_true_in_bin_tier3)} points.")
                percentile_message = "Max Price calculated using wide band quality"
            else:
                # Fallback to percentile of all points above the line (already set as default)
                if len(y_true_above) >= min_points_in_bin : # Only print if the fallback is based on a reasonable number of points
                    print(
                        f"Warning: Insufficient points above line in tiered bins for reqscore {reqscore}. "
                        f"Falling back to {percentile}th percentile of all ({len(y_true_above)}) points above line: ({fallback_percentile_price:.2f}).")

                    percentile_message = "Max Price calculated using global quality"
                elif len(y_true_above) > 0:
                     #print(
                     #   f"Warning: Insufficient points above line in tiered bins for reqscore {reqscore}. "
                     #   f"Falling back to {percentile}th percentile of all ({len(y_true_above)}) points above line (value: {fallback_percentile_price:.2f}). Number of points is low.")

                    percentile_message = "Max Price calculated using global quality"
                # If fallback_percentile_price was based on 0 points (already handled by initial check) or 1-2 points, local_percentile_price is already correctly 0 or the calculated percentile.


    # If local percentile price is 0.0 and fallback was also 0.0 due to no points above line,
    # we might want a more informative fallback. However, based on the initial check,
    # if len(y_true_above) is 0, we return 0.0 early.

    # If local_percentile_price was calculated from a bin but was 0.0 (unlikely for prices),
    # we return it. If fallback was used and was > 0.0, we return it.
    return local_percentile_price, percentile_message