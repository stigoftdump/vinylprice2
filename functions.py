import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from grid_functions import realprice
import math

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

def graph_logic(reqscore, shop_var, processed_grid):
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

    # Extract parameters for readability
    #a1, b1, c1, a_exp, b_exp, c_exp, d = params

    def predict_price_exp(quality_value):
        # Ensure it calls the correct function with the fitted params
        return sigmoid_plus_exponential(quality_value, *params)

    # Calculate predictions for all data points
    y_pred = predict_price_exp(X.flatten())

    # gets local std dev
    local_std_dev = get_stdev(y, y_pred, X, reqscore)

    # calculates prices
    predicted_price = predict_price_exp(reqscore)
    upper_bound = predicted_price + local_std_dev # Upper bound using local error estimate
    adjusted_price = predicted_price + (local_std_dev * shop_var) # Adjusted price using local error
    actual_price = realprice(adjusted_price)

    # Create a smooth curve for plotting the fitted function
    # Determine the minimum x-value for the smooth curve
    min_x_for_smooth = min(min(X.flatten()), reqscore) if X.size > 0 else reqscore

    # Determine the maximum x-value for the smooth curve
    max_x_for_smooth = max(max(X.flatten()), reqscore) if X.size > 0 else reqscore

    # Create a smooth curve for plotting the fitted function, extending to reqscore if necessary
    X_smooth = np.linspace(min_x_for_smooth, max_x_for_smooth, 200)

    y_smooth_pred = predict_price_exp(X_smooth)

    return qualities, prices, X_smooth, y_smooth_pred, predicted_price, upper_bound, actual_price

def get_stdev(y, y_pred, X, reqscore):
    # --- Calculate Local Standard Deviation instead of Global RMSE ---
    residuals = y - y_pred
    global_rmse = np.sqrt(mean_squared_error(y, y_pred))  # Keep global RMSE as a fallback

    # Define bins (e.g., by integer quality score)
    # Bins: [1, 2), [2, 3), ..., [8, 9], [9, 9+] handle edge cases
    # Or simpler: bin based on floor(quality)
    quality_floors = np.floor(X.flatten())
    req_score_floor = math.floor(reqscore)

    local_std_dev = global_rmse  # Default to global RMSE
    min_points_in_bin = 3  # Minimum points needed in a bin to calculate local std dev

    # Tier 1: Exact reqscore
    indices_in_bin_tier1 = np.where(np.isclose(X.flatten(), reqscore))[0]

    if len(indices_in_bin_tier1) >= min_points_in_bin:
        residuals_in_bin = residuals[indices_in_bin_tier1]
        local_std_dev = np.std(residuals_in_bin)
        print(
            f"Using local std dev ({local_std_dev:.2f}) for exact quality score {reqscore} based on {len(indices_in_bin_tier1)} points.")
    else:
        # Tier 2: 0.5 either side of reqscore
        lower_bound_tier2 = reqscore - 0.5
        upper_bound_tier2 = reqscore + 0.5
        indices_in_bin_tier2 = np.where((X.flatten() >= lower_bound_tier2) & (X.flatten() <= upper_bound_tier2))[0]

        if len(indices_in_bin_tier2) >= min_points_in_bin:
            residuals_in_bin = residuals[indices_in_bin_tier2]
            local_std_dev = np.std(residuals_in_bin)
            print(
                f"Using local std dev ({local_std_dev:.2f}) for quality range [{lower_bound_tier2:.2f}, {upper_bound_tier2:.2f}] based on {len(indices_in_bin_tier2)} points.")
        else:
            # Tier 3: 1.0 either side of reqscore
            lower_bound_tier3 = reqscore - 1.0
            upper_bound_tier3 = reqscore + 1.0
            indices_in_bin_tier3 = np.where((X.flatten() >= lower_bound_tier3) & (X.flatten() <= upper_bound_tier3))[0]

            if len(indices_in_bin_tier3) >= min_points_in_bin:
                residuals_in_bin = residuals[indices_in_bin_tier3]
                local_std_dev = np.std(residuals_in_bin)
                print(
                    f"Using local std dev ({local_std_dev:.2f}) for quality range [{lower_bound_tier3:.2f}, {upper_bound_tier3:.2f}] based on {len(indices_in_bin_tier3)} points.")
            else:
                # Tier 4: Fallback to global RMSE
                print(
                    f"Warning: Insufficient points in tiered bins for reqscore {reqscore}. Falling back to global RMSE ({global_rmse:.2f}).")
                local_std_dev = global_rmse
    return local_std_dev