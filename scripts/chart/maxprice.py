# used to calculate the maxprice above the calculated line of best fit.
import numpy as np

def get_points_above_line(qualities, prices, predict_func):
    """
    Identifies and extracts the data points that lie above the fitted curve.

    Args:
        qualities (list): A list of all qualities from the data.
        prices (list): A list of all inflation-adjusted prices from the data.
        predict_func (function): The function used to predict price for a given quality score.

    Returns:
        tuple: A tuple containing:
            - y_true_above_all (np.ndarray): Actual prices for points above the line.
            - X_quality_above_all (np.ndarray): Quality scores for points above the line.
            Returns empty NumPy arrays if no points are found above the line.
    """
    # numpifies the tuples
    X, y = numpify_qualities_and_prices(qualities, prices)

    y_pred = predict_func(X.flatten())

    # Flatten quality array for easier indexing
    X_quality_flat = X.flatten()

    # Identify points above the line of best fit (positive residuals)
    all_residuals = y - y_pred
    points_above_line_indices = np.where(all_residuals > 0)[0]

    # If no points are above the line in the entire dataset, return 0.0
    if len(points_above_line_indices) == 0:
        print("Warning: No sales points found above the line of best fit anywhere. Returning 0.0 percentage.")
        return np.array([]), np.array([])

    # Get all points above the line for later use
    y_true_above_all = y[points_above_line_indices]
    X_quality_above_all = X_quality_flat[points_above_line_indices]

    return y_true_above_all, X_quality_above_all

def get_percentile_price_above_line(qualities, prices, reqscore, predict_func):
    """
    Calculates a percentage representing the average of percentile price
    increases (relative to predicted price at each unique quality score) for
    points above the line within the selected quality band, using a tiered
    approach based on proximity to reqscore.

    It iterates through predefined quality tiers centered around the reqscore,
    calculating the average percentile increase for points above the fitted
    curve within each tier. The result from the first tier with sufficient
    unique quality scores is returned.

    Args:
        qualities (list): A list of all qualities from the data.
        prices (list): A list of all inflation-adjusted prices from the data.
        reqscore (float): The target quality score.
        predict_func (function): The function used to predict price for a given quality score.


    Returns:
        tuple: A tuple containing:
            - float: The calculated average percentage above the predicted line
                     for the selected band. Returns 0.0 if no points are above
                     the line or insufficient data in any tier/band.
            - float: The width value associated with the tier that was used
                     for calculation (e.g., 0.2, 0.5, 1, or 10).
    """

    # gets points above the lines
    y_true_above_all, X_quality_above_all = get_points_above_line(qualities, prices, predict_func)

    # Define the tiers with their bounds and descriptions
    tiers = [
        # (lower_bound_func, upper_bound_func, width_value)
        (lambda r: r, lambda r: r, 0.2),
        (lambda r: r - 0.5, lambda r: r + 0.5, 0.5),
        (lambda r: r - 1.0, lambda r: r + 1.0, 1),
        (lambda r: float('-inf'), lambda r: float('inf'), 10)
    ]

    # Try each tier in order
    for tier_info in tiers:
        # gets the average score for this tier
        average_percentage, width_value = get_average_percentage(tier_info, X_quality_above_all, y_true_above_all, reqscore, predict_func)

        if average_percentage is not None:
            return average_percentage, width_value

def get_average_percentage(tier_info, X_quality_above_all, y_true_above_all, reqscore, predict_func):
    """
    Calculates the average percentile price increase for points above the line
    within a specific quality band defined by a tier.

    Args:
        tier_info (tuple): A tuple (lower_bound_func, upper_bound_func, width_value)
                           defining the quality band relative to reqscore.
        X_quality_above_all (np.ndarray): Quality scores for all points above the line.
        y_true_above_all (np.ndarray): Actual prices for all points above the line.
        reqscore (float): The target quality score.
        predict_func (function): The function used to predict price for a given quality score.
        percentile (float): The desired percentile to calculate (0 to 100).

    Returns:
        float or None: The calculated average percentage increase for the points
                       in the band, or None if there are insufficient unique
                       quality scores with data in this band.
    """
    # Unpack tier information
    lower_bound_func, upper_bound_func, width_value = tier_info
    lower_bound = lower_bound_func(reqscore)
    upper_bound = upper_bound_func(reqscore)

    # Filter points within this tier's bounds
    indices_in_band = np.where(
        (X_quality_above_all >= lower_bound) & (X_quality_above_all <= upper_bound)
    )[0]

    y_true_in_band = y_true_above_all[indices_in_band]
    X_quality_in_band = X_quality_above_all[indices_in_band]

    average_percentage = calculate_average_percentage_for_subset(y_true_in_band, X_quality_in_band, predict_func)

    return average_percentage, width_value

def calculate_average_percentage_for_subset(actual_prices, quality_scores, predict_func):
    """
    Helper function to calculate the average percentage increase across unique
    quality scores within a given subset of data points (assumed to be above the line).

    For each unique quality score in the subset, it calculates the specified
    percentile price and determines its percentage increase relative to the
    predicted price at that score. It then returns the average of these
    percentage increases across all unique scores that meet the minimum point
    requirement.

    Args:
        actual_prices (np.ndarray): Actual price values for the subset of points.
        quality_scores (np.ndarray): Quality scores corresponding to actual_prices.
        predict_func (function): The function used to predict price for a given quality score.
        percentile (float): The desired percentile to calculate (0 to 100).

    Returns:
        float or None: The calculated average percentage increase for the subset,
                       or None if there are insufficient unique quality scores
                       with enough data points.
    """
    # defines constants
    MIN_POINTS_FOR_PERCENTILE_PER_SCORE = 1  # Minimum points needed at a specific score for percentile
    MIN_SCORES_FOR_AVERAGE = 1  # Minimum unique quality scores with valid percentiles
    percentile = 90

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

def numpify_qualities_and_prices(qualities, prices):
    """
    Converts lists of qualities and prices into NumPy arrays suitable for fitting.

    Args:
        qualities (list): List of quality scores.
        prices (list): List of prices.

    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): 2D NumPy array of quality scores (shape: (n_samples, 1)).
            - y (np.ndarray): 1D NumPy array of prices (shape: (n_samples,)).
    """
    X = np.array(qualities).reshape(-1, 1)  # Quality as independent variable
    y = np.array(prices)  # Price as dependent variable
    return X, y