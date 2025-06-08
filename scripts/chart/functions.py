import numpy as np
from scipy.optimize import curve_fit
from scripts.grid.grid_functions import realprice
from .maxprice import get_percentile_price_above_line, numpify_qualities_and_prices

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
    """

    X, y = numpify_qualities_and_prices(qualities, prices)

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

    return params, predict_func

def get_actual_price(reqscore, shop_var, qualities, prices, predicted_price, predict_func):
    """
    Calculates the upper bound and final adjusted price based on the predicted
    price and percentile analysis of points above the fitted curve.

    Args:
        reqscore (float): The target quality score.
        shop_var (float): A factor applied to the calculated price difference
                          between the upper bound and the predicted price.
        qualities (list): A list of all qualities from the data.
        prices (list): A list of all inflation-adjusted prices from the data.
        predicted_price (float): The predicted price for the reqscore
                                 (calculated separately).

    Returns:
        tuple: A tuple containing:
            - upper_bound (float): The calculated upper bound price based on percentile analysis.
            - actual_price (float): The final rounded price after applying shop_var.
            - search_width (float): The width of the quality band used for the percentile calculation.
    """

    # gets percentile price above line
    percentage_above_line, search_width = get_percentile_price_above_line(qualities, prices, reqscore, predict_func)
    # calculates prices
    upper_bound = predicted_price * ((percentage_above_line/100)+1)
    adjusted_price = predicted_price + ((upper_bound - predicted_price) * shop_var)
    actual_price = realprice(float(adjusted_price))

    return upper_bound, actual_price, search_width

def predict_price(reqscore, predict_func):
    """
    Predicts the price for a given required quality score using the fitted curve.

    Args:
        qualities (list): A list of all qualities from the data.
        prices (list): A list of all inflation-adjusted prices from the data.
        reqscore (float): The target quality score for prediction.
        predict_func (function): The function used to predict price for a given quality score.


    Returns:
        float: The predicted price at the reqscore based on the fitted curve.
               Returns 0.0 if curve fitting fails.
    """
    # gets the price from the predict function
    predicted_price = predict_func(reqscore)

    return predicted_price

def generate_smooth_curve_data(qualities, prices, reqscore, predict_func):
    """
    Create a smooth curve for plotting the fitted function

    Args:
        qualities (list): A list of all  qualities
        prices (list): A list of all prices
        reqscore (float): The required score, only used if the reqscore is outside of bounds for the qualities
        predict_func (function): The function used to predict price for a given quality score.

    Returns:
        tuple: A tuple containing:
            X-smooth (numparray): Smoothed X values for the line of best fit
            y_smooth_prted (numparray): Smoothed Y values for the line of best fit

    """
    # numpifies the qualities and prices
    X, y = numpify_qualities_and_prices(qualities, prices)

    # Determine the minimum x-value for the smooth curve
    min_x_for_smooth = min(min(X.flatten()), reqscore) if X.size > 0 else reqscore

    # Determine the maximum x-value for the smooth curve
    max_x_for_smooth = max(max(X.flatten()), reqscore) if X.size > 0 else reqscore

    # Create a smooth curve for plotting the fitted function, extending to reqscore if necessary
    X_smooth = np.linspace(min_x_for_smooth, max_x_for_smooth, 200)

    y_smooth_pred = predict_func(X_smooth)

    return X_smooth, y_smooth_pred

def write_output(api_data, predicted_price, upper_bound, actual_price, chart_data):
    """
    Writes the outputs data from the bits of code
    """

    output_data = {}

    # Store API data in output_data if available
    if api_data:
        output_data["api_artist"] = api_data.get("api_artist")
        output_data["api_title"] = api_data.get("api_title")
        output_data["api_year"] = api_data.get("api_year")
        output_data["api_original_year"] = api_data.get("api_original_year")
        # You can add more API fields here if needed later
        if api_data.get("api_artist") or api_data.get("api_title"):
            print(f"VIN.PY: API Data processed: {api_data.get('api_artist', '')} - {api_data.get('api_title', '')}")

    # If all calculations are successful, populate output_data
    output_data["calculated_price"] = round(predicted_price, 2)
    output_data["upper_bound"] = round(upper_bound, 2)
    output_data["actual_price"] = round(actual_price, 2)
    output_data["chart_data"] = chart_data

    return output_data

