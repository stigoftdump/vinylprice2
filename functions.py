import numpy as np
import sys
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from grid_functions import realprice
import json

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

def return_variables(argument):
    # returns the variables from the argument sent in sys
    # --- MODIFY Check ---
    if len(argument) < 7: # Now expect at least 8 arguments (script name + 6 data args)
    # --- END MODIFY ---
        print("Error: data is missing.")
        # Output JSON error for Flask to potentially catch
        print(json.dumps({"status_message": "Error: Missing arguments for script."}))
        sys.exit(1) # Still exit, but Flask might see the JSON message now
    try:
        reqscore = float(argument[1])
        shop_var = float(argument[2])
        start_date = argument[3]
        add_data = argument[4]
        discogs_data = argument[5]
        # --- ADD THIS ---
        points_to_delete_json = argument[6] # Get the new argument
        # --- END ADD ---

    except ValueError:
        print("Error: Both reqscore and shop_var must be numbers.")
         # Output JSON error
        print(json.dumps({"status_message": "Error: reqscore and shop_var must be numbers."}))
        sys.exit(1)
    except IndexError:
         print("Error: Not enough arguments provided.")
         # Output JSON error
         print(json.dumps({"status_message": "Error: Not enough arguments provided to script."}))
         sys.exit(1)


    # --- MODIFY Return ---
    return reqscore, shop_var, start_date, add_data, discogs_data, points_to_delete_json # Return the new variable
    # --- END MODIFY ---

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

    return qualities, prices, X_smooth, y_smooth_pred, reqscore, predicted_price, upper_bound, actual_price