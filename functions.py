import numpy as np
from scipy.optimize import curve_fit
from grid_functions import realprice
import pickle
import statsmodels.api as sm

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

def get_actual_price(reqscore, shop_var, qualities, prices, predicted_price):
    """
    Performs curve fitting and price prediction based on processed sales data.

    Fits a sigmoid-plus-exponential model to the quality vs. price data,
    calculates predicted prices, error bounds, and generates data for plotting.

    Args:
        reqscore (float): The target quality score for which to predict the price.
        shop_var (float): A factor applied to the calculated upper price bound.
        qualities (list): A list of all  qualities
        prices (list): A list of all prices

    Returns:
        tuple: A tuple containing:
            - predicted_price (float): The predicted price for the reqscore based on the fitted curve.
            - upper_bound (float): The calculated upper bound price based on percentiles.
            - actual_price (float): The final price after applying shop_var and rounding.
    """

    # new percentile code
    pi_details = get_prediction_interval_details(qualities, prices, reqscore, confidence_level=0.90)

    upper_bound = 0.0 # Initialize upper_bound

    if pi_details['error']:
        print(f"Warning: Error calculating prediction interval: {pi_details['error']}")
        # Fallback if prediction interval calculation fails
        # You could adjust this fallback as needed, e.g., a fixed percentage
        upper_bound = predicted_price * 1.15 # Fallback to 15% above predicted if PI fails
    else:
        # Use the upper bound from the linear model's prediction interval
        upper_bound = pi_details['upper_prediction_interval']
        # You might want to ensure the upper_bound is not less than predicted_price,
        # especially if the linear model diverges significantly from the non-linear one.
        upper_bound = max(upper_bound, predicted_price)

    # calculates prices
    adjusted_price = predicted_price + ((upper_bound - predicted_price) * shop_var)
    actual_price = realprice(float(adjusted_price))

    return upper_bound, actual_price

def numpify_qualities_and_prices(qualities, prices):
    X = np.array(qualities).reshape(-1, 1)  # Quality as independent variable
    y = np.array(prices)  # Price as dependent variable
    return X, y

def predict_price(qualities, prices, reqscore):
    # Get the fitted parameters and prediction function
    params, predict_func = fit_curve_and_get_params(qualities, prices)

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

    params, predict_func = fit_curve_and_get_params(qualities, prices)

    X, y = numpify_qualities_and_prices(qualities, prices)

    # Determine the minimum x-value for the smooth curve
    min_x_for_smooth = min(min(X.flatten()), reqscore) if X.size > 0 else reqscore

    # Determine the maximum x-value for the smooth curve
    max_x_for_smooth = max(max(X.flatten()), reqscore) if X.size > 0 else reqscore

    # Create a smooth curve for plotting the fitted function, extending to reqscore if necessary
    X_smooth = np.linspace(min_x_for_smooth, max_x_for_smooth, 200)

    y_smooth_pred = predict_func(X_smooth)

    return X_smooth, y_smooth_pred

def get_prediction_interval_details(qualities, prices, target_quality, confidence_level=0.95):
    """
    Calculates the predicted price and the upper bound of the prediction interval
    for a given target quality, using linear regression. Also returns the percentage
    difference between the upper prediction interval bound and the predicted price.

    Using linear regression than than tha actual non-liner regression is a compromise that means that we
    can use the statsmodels to derive the

    Args:
        qualities (list or np.array): List of quality scores (independent variable, X).
        prices (list or np.array): List of corresponding prices (dependent variable, Y).
        target_quality (float): The specific quality score for which to make the prediction.
        confidence_level (float, optional): The desired confidence level for the
                                           prediction interval (e.g., 0.95 for 95%).
                                           Defaults to 0.95.

    Returns:
        dict: A dictionary containing:
            'predicted_price_on_line' (float): The price predicted by the regression line for target_quality.
            'lower_prediction_interval' (float): The lower bound of the prediction interval.
            'upper_prediction_interval' (float): The upper bound of the prediction interval.
            'percentage_over_predicted' (float): The percentage that the upper_prediction_interval
                                                 is above the predicted_price_on_line.
                                                 Returns float('inf') if predicted_price_on_line is 0.
            'regression_summary' (str): A summary of the OLS regression (can be None if model fails).
            'error' (str or None): An error message if calculations fail, otherwise None.
    """
    results = {
        'predicted_price_on_line': None,
        'lower_prediction_interval': None,
        'upper_prediction_interval': None,
        'percentage_over_predicted': None,
        'regression_summary': None,
        'error': None
    }

    if len(qualities) < 2 or len(prices) < 2 or len(qualities) != len(prices):
        results['error'] = "Insufficient data or mismatched lengths for qualities and prices."
        return results

    try:

        # turns into numpty arrays
        X_all, y_all = numpify_qualities_and_prices(qualities, prices)

        # 1. Fit an initial linear model to all data to get initial residuals
        X_all_with_const = sm.add_constant(X_all)
        model_all = sm.OLS(y_all, X_all_with_const)
        fitted_model_all = model_all.fit()
        y_pred_all = fitted_model_all.predict(X_all_with_const)

        # 2. Identify data points where actual price is higher than predicted price
        positive_residuals_indices = np.where(y_all > y_pred_all)[0]

        if len(positive_residuals_indices) < 2:
            results['error'] = "Insufficient data points where actual price is above the predicted linear price to calculate prediction interval. Need at least 2 such points."
            return results

        # 3. Filter data to only include points with positive residuals
        X_filtered = X_all[positive_residuals_indices]
        y_filtered = y_all[positive_residuals_indices]

        # Prepare filtered data for statsmodels (add constant for intercept)
        X_filtered_with_const = sm.add_constant(X_filtered)

        # 4. Fit OLS regression model *only on the filtered data*
        model = sm.OLS(y_filtered, X_filtered_with_const)
        fitted_model = model.fit()
        results['regression_summary'] = str(fitted_model.summary())

        # Value for which to predict (must include constant)
        target_X_with_const = np.array([1, target_quality])

        # Get prediction results object from the model fitted on filtered data
        prediction = fitted_model.get_prediction(target_X_with_const)

        # Prediction intervals are for a *new observation*
        # The alpha for conf_int is 1 - confidence_level
        alpha = 1 - confidence_level
        pred_interval = prediction.conf_int(obs=True, alpha=alpha) # obs=True for prediction interval

        results['predicted_price_on_line'] = prediction.predicted_mean[0]
        results['lower_prediction_interval'] = pred_interval[0, 0]
        results['upper_prediction_interval'] = pred_interval[0, 1]

        if results['predicted_price_on_line'] is not None and results['predicted_price_on_line'] != 0:
            results['percentage_over_predicted'] = \
                ((results['upper_prediction_interval'] - results['predicted_price_on_line']) /
                 results['predicted_price_on_line']) * 100
        elif results['predicted_price_on_line'] == 0 and results['upper_prediction_interval'] is not None:
            results['percentage_over_predicted'] = float('inf') # Or handle as appropriate

    except Exception as e:
        results['error'] = f"Error during prediction interval calculation: {str(e)}"
        # You might want to print the full traceback for debugging
        # import traceback
        # print(traceback.format_exc())
    return results