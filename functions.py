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

    predicted_price = predict_price_exp(reqscore)

    # gets percentile price above line
    # NOTE: The shop_var is now applied to this percentile price
    percentage_above_line, percentile_message = get_percentile_price_above_line(y, y_pred, X, reqscore,
                                                                                predicted_price_at_reqscore=predicted_price,
                                                                                predict_func=predict_price_exp,
                                                                                percentile=90)

    # calculates prices
    upper_bound = predicted_price * ((percentage_above_line/100)+1)
    adjusted_price = predicted_price + ((upper_bound - predicted_price) * shop_var)
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

# Signature remains the same as previous version that accepted predict_func
def get_percentile_price_above_line(y_true, y_pred, X_quality, reqscore, predicted_price_at_reqscore, predict_func, percentile=90):
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
    all_residuals = y_true - y_pred
    X_quality_flat = X_quality.flatten() # Flatten quality array for easier indexing

    # Identify points above the line of best fit (positive residuals)
    points_above_line_indices = np.where(all_residuals > 0)[0]

    # If no points are above the line in the entire dataset, return 0.0
    if len(points_above_line_indices) == 0:
        print("Warning: No sales points found above the line of best fit anywhere. Returning 0.0 percentage.")
        return 0.0, "Max Price %: No points above line in dataset"

    y_true_above_all = y_true[points_above_line_indices]
    X_quality_flat_above_all = X_quality_flat[points_above_line_indices]

    min_points_for_percentile_per_score = 3 # Minimum points needed at a specific unique score for its percentile
    min_scores_for_average = 1 # Minimum unique quality scores with valid percentiles to calculate the overall average

    # Helper function to calculate the average percentage increase across unique quality scores within a subset
    # This helper operates on the subset of points *already identified as being above the line*
    def calculate_average_percentage_for_subset(actual_prices_subset_above_line, quality_scores_subset_above_line, predict_func, percentile):
        # Ensure the subset is not empty (already handled by outer check, but good for helper robustness)
        if len(actual_prices_subset_above_line) == 0:
            return None

        unique_quality_scores_in_subset_above_line = np.unique(quality_scores_subset_above_line)
        percentage_increases_per_unique_score_in_subset = []

        for unique_score in unique_quality_scores_in_subset_above_line:
            # Find points within this subset *above the line* that match the current unique score
            indices_at_unique_score_in_subset_above_line = np.where(np.isclose(quality_scores_subset_above_line, unique_score))[0]
            subset_actual_prices_at_score_above_line = actual_prices_subset_above_line[indices_at_unique_score_in_subset_above_line]

            if len(subset_actual_prices_at_score_above_line) >= min_points_for_percentile_per_score:
                # Calculate 90th percentile price for points at this unique score *above the line* within the subset
                percentile_price_at_score = np.percentile(subset_actual_prices_at_score_above_line, percentile)

                # Calculate the predicted price at this unique quality score
                predicted_price_at_score = predict_func(unique_score)

                # Calculate percentage increase for this unique s core band
                percentage_increase_at_score = 0.0
                if predicted_price_at_score > 0:
                    percentage_increase_at_score = (
                        (percentile_price_at_score - predicted_price_at_score) / predicted_price_at_score
                    ) * 100.0
                    # Ensure the percentage is not negative
                    percentage_increase_at_score = max(0.0, percentage_increase_at_score)
                # If predicted_price_at_score is 0 or negative, percentage_increase_at_score remains 0.0


                percentage_increases_per_unique_score_in_subset.append(percentage_increase_at_score)

        # If we have enough unique quality scores with valid percentiles, calculate the average
        if len(percentage_increases_per_unique_score_in_subset) >= min_scores_for_average:
            average_percentage = np.mean(percentage_increases_per_unique_score_in_subset)
            return average_percentage
        else:
            # Not enough unique quality scores with sufficient points above the line for percentile calculation in this subset
            return None


    # --- Tier 1: Exact reqscore ---
    # Filter points above the line (y_true_above_all, X_quality_flat_above_all) to find those exactly at reqscore
    indices_at_reqscore_above = np.where(np.isclose(X_quality_flat_above_all, reqscore))[0]
    y_true_at_reqscore_above = y_true_above_all[indices_at_reqscore_above]
    X_quality_at_reqscore_above = X_quality_flat_above_all[indices_at_reqscore_above] # Get quality scores for this subset
    average_percentage = calculate_average_percentage_for_subset(y_true_at_reqscore_above, X_quality_at_reqscore_above, predict_func, percentile)
    if average_percentage is not None:
        print(f"Using exact reqscore {reqscore} ({len(y_true_at_reqscore_above)} points above line in total in this tier) for average percentage calculation.")
        # Count how many unique scores within this tier's points above the line contributed to the average for the message
        unique_scores_contributing = np.unique(X_quality_at_reqscore_above[[i for i in range(len(X_quality_at_reqscore_above)) if len(np.where(np.isclose(X_quality_at_reqscore_above, X_quality_at_reqscore_above[i]))[0]) >= min_points_for_percentile_per_score]])
        return average_percentage, f"Max Price %: Avg across {len(unique_scores_contributing)} unique scores at exact quality ({reqscore})"


    # --- Tier 2: 0.5 Band around reqscore ---
    lower_bound_tier2 = reqscore - 0.5
    upper_bound_tier2 = reqscore + 0.5
    # Filter points above the line (y_true_above_all, X_quality_flat_above_all) to find those within the 0.5 band
    indices_in_band_tier2_above = np.where(
        (X_quality_flat_above_all >= lower_bound_tier2) & (X_quality_flat_above_all <= upper_bound_tier2)
    )[0]
    y_true_in_band_tier2_above = y_true_above_all[indices_in_band_tier2_above]
    X_quality_in_band_tier2_above = X_quality_flat_above_all[indices_in_band_tier2_above] # Get quality scores for this subset
    average_percentage = calculate_average_percentage_for_subset(y_true_in_band_tier2_above, X_quality_in_band_tier2_above, predict_func, percentile)
    if average_percentage is not None:
        print(f"Using 0.5 band around reqscore ({len(y_true_in_band_tier2_above)} points above line in total in this tier) for average percentage calculation.")
        # Count how many unique scores within this tier's points above the line contributed to the average for the message
        unique_scores_contributing = np.unique(X_quality_in_band_tier2_above[[i for i in range(len(X_quality_in_band_tier2_above)) if len(np.where(np.isclose(X_quality_in_band_tier2_above, X_quality_in_band_tier2_above[i]))[0]) >= min_points_for_percentile_per_score]])
        return average_percentage, f"Max Price %: Avg across {len(unique_scores_contributing)} unique scores in [{lower_bound_tier2:.2f}, {upper_bound_tier2:.2f}] band"

    # --- Tier 3: 1.0 Band around reqscore ---
    lower_bound_tier3 = reqscore - 1.0
    upper_bound_tier3 = reqscore + 1.0
    # Filter points above the line (y_true_above_all, X_quality_flat_above_all) to find those within the 1.0 band
    indices_in_band_tier3_above = np.where(
        (X_quality_flat_above_all >= lower_bound_tier3) & (X_quality_flat_above_all <= upper_bound_tier3)
    )[0]
    y_true_in_band_tier3_above = y_true_above_all[indices_in_band_tier3_above]
    X_quality_in_band_tier3_above = X_quality_flat_above_all[indices_in_band_tier3_above] # Get quality scores for this subset
    average_percentage = calculate_average_percentage_for_subset(y_true_in_band_tier3_above, X_quality_in_band_tier3_above, predict_func, percentile)
    if average_percentage is not None:
        print(f"Using 1.0 band around reqscore ({len(y_true_in_band_tier3_above)} points above line in total in this tier) for average percentage calculation.")
        # Count how many unique scores within this tier's points above the line contributed to the average for the message
        unique_scores_contributing = np.unique(X_quality_in_band_tier3_above[[i for i in range(len(X_quality_in_band_tier3_above)) if len(np.where(np.isclose(X_quality_in_band_tier3_above, X_quality_in_band_tier3_above[i]))[0]) >= min_points_for_percentile_per_score]])
        return average_percentage, f"Max Price %: Avg across {len(unique_scores_contributing)} unique scores in [{lower_bound_tier3:.2f}, {upper_bound_tier3:.2f}] band"


    # --- Tier 4: All points above line ---
    # Use all points that were initially identified as being above the line
    average_percentage = calculate_average_percentage_for_subset(y_true_above_all, X_quality_flat_above_all, predict_func, percentile) # Use all points and their qualities
    if average_percentage is not None:
        print(f"Using all points above line ({len(y_true_above_all)} points in total in this tier) for average percentage calculation.")
         # Count how many unique scores within this tier's points above the line contributed to the average for the message
        unique_scores_contributing = np.unique(X_quality_flat_above_all[[i for i in range(len(X_quality_flat_above_all)) if len(np.where(np.isclose(X_quality_flat_above_all, X_quality_flat_above_all[i]))[0]) >= min_points_for_percentile_per_score]])
        return average_percentage, f"Max Price %: Avg across {len(unique_scores_contributing)} unique scores from all points above line"

    # --- Fallback: Insufficient data in any tier/band for the average percentile calculation ---
    print("Warning: Insufficient points above line in any relevant unique quality score bands within tiers for average percentile calculation. Returning 0.0 percentage.")
    return 0.0, "Max Price %: Insufficient unique quality data in bands"
