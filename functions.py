import numpy as np
from scipy.optimize import curve_fit
from grid_functions import realprice
import pickle

unified_save_file = "vinylpricedata.pkl"

def read_application_data():
    """Reads the entire data dictionary from the unified pickle file."""
    try:
        with open(unified_save_file, 'rb') as f:
            data = pickle.load(f)
            return data
    except (FileNotFoundError, EOFError, pickle.UnpicklingError): # Handle empty or corrupted file
        return {} # Return an empty dict if file not found or unpickling error

def write_application_data(data):
    """Writes the entire data dictionary to the unified pickle file."""
    try:
        with open(unified_save_file, 'wb') as f:
            pickle.dump(data, f)
    except IOError as e:
        print(f"Error writing to {unified_save_file}: {e}")

def read_save_value(datatype, default):
    """
    Reads the data type from the saved file

    It returns the default if it cant read the file

    """
    data = read_application_data()
    datareturn = data.get(datatype, default)

    return datareturn

def write_save_value(value, datatype):
    """
    Write the value to the datatype in the save file
    """
    data = read_application_data() # Read existing data
    data[datatype] = value
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

def fit_curve_and_get_params(qualities, prices):
    """
    Performs curve fitting using the sigmoid_plus_exponential model
    and returns the fitted parameters and a prediction function.

    Args:
        qualities (list): Original quality scores from the data.
        prices (list): Original (inflation-adjusted) prices from the data.

    Returns:
        tuple: A tuple containing:
            - params (np.ndarray): The optimal parameters for the fitted curve.
            - predict_func (function): A function that takes a quality value and returns
                                       the predicted price based on the fitted parameters.
            - X (np.ndarray): The quality scores converted to a numpy array for fitting.
            - y (np.ndarray): The prices converted to a numpy array for fitting.
    """
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

    # Fit the sigmoid_plus_exponential model
    params, _ = curve_fit(
        sigmoid_plus_exponential,
        X.flatten(),
        y,
        p0=initial_guess,
        bounds=bounds,
        maxfev=100000
    )

    def predict_func(quality_value):
        """Predicts price using the fitted sigmoid_plus_exponential model."""
        return sigmoid_plus_exponential(quality_value, *params)

    return params, predict_func, X, y

def graph_logic(reqscore, shop_var, qualities, prices):
    """
    Performs curve fitting and price prediction based on processed sales data.

    Fits a sigmoid-plus-exponential model to the quality vs. price data,
    calculates predicted prices, error bounds, and generates data for plotting.

    Args:
        reqscore (float): The target quality score for which to predict the price.
        shop_var (float): A factor applied to the calculated upper price bound.
        qualities (list): A list ofall  qualities
        prices (list): A list of all prices

    Returns:
        tuple: A tuple containing:
            - predicted_price (float): The predicted price for the reqscore based on the fitted curve.
            - upper_bound (float): The calculated upper bound price based on percentiles.
            - actual_price (float): The final price after applying shop_var and rounding.
    """
    # function that returns everything needed to make a chart

    # Get the fitted parameters and prediction function
    params, predict_func, X, y = fit_curve_and_get_params(qualities, prices)

    y_pred = predict_func(X.flatten())

    predicted_price = predict_func(reqscore)

    # gets percentile price above line
    # NOTE: The shop_var is now applied to this percentile price
    percentage_above_line, percentile_message, search_width = get_percentile_price_above_line(y, y_pred, X, reqscore,
                                                                                predict_func=predict_func,
                                                                                percentile=90)
    # calculates prices
    upper_bound = predicted_price * ((percentage_above_line/100)+1)
    adjusted_price = predicted_price + ((upper_bound - predicted_price) * shop_var)
    actual_price = realprice(float(adjusted_price))

    return upper_bound, actual_price, percentile_message, search_width

def predict_price(qualities, prices, reqscore):
    # Get the fitted parameters and prediction function
    params, predict_func, X, y = fit_curve_and_get_params(qualities, prices)

    # gets the price from the predict function
    predicted_price = predict_func(reqscore)

    return predicted_price

def generate_smooth_curve_data(qualities, prices, reqscore):
    """
    Create a smooth curve for plotting the fitted function

    Args:
        qualities (list): A list of all  qualities
        prices (list): A list of all prices
        reqscore (float): The required score, only used if the reqscore is outside of bounds for the qualities

    Returns:
        tuple: A tuple containing:
            X-smooth (numparray): Smoothed X values for the line of best fit
            y_smooth_prted (numparray): Smoothed Y values for the line of best fit

    """

    params, predict_func, X, y = fit_curve_and_get_params(qualities, prices)

    # Determine the minimum x-value for the smooth curve
    min_x_for_smooth = min(min(X.flatten()), reqscore) if X.size > 0 else reqscore

    # Determine the maximum x-value for the smooth curve
    max_x_for_smooth = max(max(X.flatten()), reqscore) if X.size > 0 else reqscore

    # Create a smooth curve for plotting the fitted function, extending to reqscore if necessary
    X_smooth = np.linspace(min_x_for_smooth, max_x_for_smooth, 200)

    y_smooth_pred = predict_func(X_smooth)

    return X_smooth, y_smooth_pred

# Signature remains the same as previous version that accepted predict_func
def get_percentile_price_above_line(y_true, y_pred, X_quality, reqscore, predict_func,
                                    percentile=90):
    """
    Calculates a percentage representing the average of percentile price
    increases (relative to predicted price at each unique quality score) for
    points above the line within the selected quality band, using a tiered
    approach based on proximity to reqscore.

    Args:
        y_true (np.ndarray): The actual price values.
        y_pred (np.ndarray): The predicted price values from the model.
        X_quality (np.ndarray): The quality scores corresponding to y_true/y_pred.
                                Expected to be a 2D array from reshape(-1,1).
        reqscore (float): The target quality score.
        predicted_price_at_reqscore (float): The predicted price at the reqscore (kept for context/messaging).
        predict_func (function): The function used to predict price for a given quality score.
        percentile (float): The desired percentile to calculate (0 to 100).

    Returns:
        tuple: A tuple containing:
            - float: The calculated average percentage above the predicted line for the selected band.
                     Returns 0.0 if no points are above the line or insufficient data in any tier/band.
            - str: A message indicating which tier was used for calculation.
    """

    # Constants
    MIN_POINTS_FOR_PERCENTILE_PER_SCORE = 3  # Minimum points needed at a specific score for percentile
    MIN_SCORES_FOR_AVERAGE = 1  # Minimum unique quality scores with valid percentiles

    # Flatten quality array for easier indexing
    X_quality_flat = X_quality.flatten()

    # Identify points above the line of best fit (positive residuals)
    all_residuals = y_true - y_pred
    points_above_line_indices = np.where(all_residuals > 0)[0]

    # If no points are above the line in the entire dataset, return 0.0
    if len(points_above_line_indices) == 0:
        print("Warning: No sales points found above the line of best fit anywhere. Returning 0.0 percentage.")
        return 0.0, "Max Price %: No points above line in dataset"

    # Get all points above the line for later use
    y_true_above_all = y_true[points_above_line_indices]
    X_quality_above_all = X_quality_flat[points_above_line_indices]

    def calculate_average_percentage_for_subset(actual_prices, quality_scores):
        """Helper function to calculate the average percentage increase across unique quality scores."""
        if len(actual_prices) == 0:
            return None

        unique_scores = np.unique(quality_scores)
        percentage_increases = []

        for score in unique_scores:
            # Find points within this subset above the line that match the current score
            indices = np.where(np.isclose(quality_scores, score))[0]
            prices_at_score = actual_prices[indices]

            if len(prices_at_score) >= MIN_POINTS_FOR_PERCENTILE_PER_SCORE:
                # Calculate percentile price for points at this score above the line
                percentile_price = np.percentile(prices_at_score, percentile)
                predicted_price = predict_func(score)

                # Calculate percentage increase for this score
                percentage_increase = 0.0
                if predicted_price > 0:
                    percentage_increase = max(0.0, ((percentile_price - predicted_price) / predicted_price) * 100.0)

                percentage_increases.append(percentage_increase)

        # Return average if we have enough scores with valid percentiles
        if len(percentage_increases) >= MIN_SCORES_FOR_AVERAGE:
            return np.mean(percentage_increases)
        return None

    # Define the tiers with their bounds and descriptions
    tiers = [
        # (lower_bound_func, upper_bound_func, description_func, width_value)
        (lambda r: r, lambda r: r, lambda n, r: f"Max Price calculated using {n} prices at score {round(r, 2)}", 0.2),
        (lambda r: r - 0.5, lambda r: r + 0.5, lambda n, lb,
                                                      ub: f"Max Price calculated using {n} prices between scores {round(lb, 2):.2f} and {round(ub, 2):.2f}", 0.5),
        (lambda r: r - 1.0, lambda r: r + 1.0, lambda n, lb,
                                                      ub: f"Max Price calculated using {n} prices between scores {round(lb, 2):.2f} and {round(ub, 2):.2f}", 1),
        (lambda r: float('-inf'), lambda r: float('inf'),
         lambda n: f"Max Price calculated using {n} prices across all scores", 10)
    ]

    # Try each tier in order
    for tier_index, tier_bounds in enumerate(tiers):
        if tier_index < 3:  # For the first three tiers with specific bounds
            lower_bound = tier_bounds[0](reqscore)
            upper_bound = tier_bounds[1](reqscore)

            # Filter points within this tier's bounds
            indices_in_band = np.where(
                (X_quality_above_all >= lower_bound) & (X_quality_above_all <= upper_bound)
            )[0]

            y_true_in_band = y_true_above_all[indices_in_band]
            X_quality_in_band = X_quality_above_all[indices_in_band]

            average_percentage = calculate_average_percentage_for_subset(y_true_in_band, X_quality_in_band)

            if average_percentage is not None:
                tier_type = "exact reqscore" if tier_index == 0 else f"{0.5 if tier_index == 1 else 1.0} band around reqscore"
                print(
                    f"Using {tier_type} ({len(y_true_in_band)} points above line in total in this tier) for average percentage calculation.")

                # Generate appropriate message based on tier
                if tier_index == 0:
                    message = tier_bounds[2](len(y_true_in_band), reqscore)
                else:
                    message = tier_bounds[2](len(y_true_in_band), lower_bound, upper_bound)

                # gets the width value
                width_value = tier_bounds[3]

                return average_percentage, message, width_value
        else:
            # Last tier - use all points above the line
            average_percentage = calculate_average_percentage_for_subset(y_true_above_all, X_quality_above_all)

            if average_percentage is not None:
                print(
                    f"Using all points above line ({len(y_true_above_all)} points in total in this tier) for average percentage calculation.")
                message = tier_bounds[2](len(y_true_above_all))
                width_value = tier_bounds[3]
                return average_percentage, message, width_value

    # Fallback: Insufficient data in any tier
    print(
        "Warning: Insufficient points above line in any relevant unique quality score bands within tiers for average percentile calculation. Returning 0.0 percentage.")
    return 0.0, "Max Price %: Insufficient unique quality data in bands"
