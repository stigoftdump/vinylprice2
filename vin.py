from functions import graph_logic, read_save_value, write_save_value, generate_smooth_curve_data
from grid_functions import make_processed_grid, delete_points, extract_tuples

def calculate_vin_data(reqscore, shop_var, start_date, add_data, discogs_data, points_to_delete_json):
    """
    Main calculation engine.

    Orchestrates the process of parsing input data, optionally merging with saved data,
    deleting selected points, performing graph fitting and price prediction,
    and preparing the output data including chart information.

    Args:
        reqscore (float): The target quality score for price prediction.
        shop_var (float): The shop variance factor to apply to the price uncertainty.
        start_date (str): The start date ('YYYY-MM-DD') for filtering sales data.
        add_data (str): String ('True' or 'False') indicating whether to merge
                            current data with previously saved data.
        discogs_data (str): Raw text data pasted by the user (Discogs sales history).
        points_to_delete_json (str): JSON string array of points selected for deletion.

    Returns:
        dict: A dictionary containing the calculated results:
              - calculated_price (float or None): Predicted price before adjustments.
              - upper_bound (float or None): Predicted price + standard deviation.
              - actual_price (float or None): Final rounded price after shop_var adjustment.
              - status_message (str): A message indicating success or errors during processing.
              - chart_data (dict): Data structured for generating the chart via Chart.js.
                                   Empty if calculation fails early.
    """
    status_message = None
    info_message = None
    error_message = None

    # if discogs data is empty, just load the saves processed_grid and use that, otherwise do the whole thing.
    if not discogs_data:
        # load the saved file
        processed_grid = read_save_value("processed_grid", {})
    else:
        # Gets the processed_grid from the discogs_data sent over
        processed_grid, status_message = make_processed_grid(discogs_data, start_date)

    # deletes points if needed
    processed_grid, deleted_count = delete_points(points_to_delete_json, processed_grid)

    # If add_data is True, load the previously saved processed_grid and add it to the current one
    if add_data == "True" and discogs_data:
        saved_processed_grid = read_save_value("processed_grid", {})
        processed_grid.extend(saved_processed_grid) # Add saved data *after* potential deletion

    # Check again if the grid is empty after deleting or loading
    if not processed_grid:
         error_message = "No data points available for analysis."
         output_data = {"calculated_price": None, "upper_bound": None, "actual_price": None, "error_message": status_message, "chart_data": {}}
         return output_data

    # Save processed grid to file
    write_save_value(processed_grid, "processed_grid")

    # Extracts elements
    qualities, prices, dates, comments = extract_tuples(processed_grid)

    # graph logic to get the variables for the output
    # Make sure graph_logic handles potentially empty lists gracefully if grid becomes empty
    try:
        predicted_price, upper_bound, actual_price, percentile_message, search_width = graph_logic(reqscore, shop_var, qualities, prices)
    except Exception as e:
        # Handle potential errors in graph_logic if the grid is unusual after filtering
        error_message = f"Error during graph calculation: {e}"
        output_data = {"calculated_price": None, "upper_bound": None, "actual_price": None, "error_message": error_message, "chart_data": {}}
        return output_data

    # Gets the smoothed data for the chart
    X_smooth, y_smooth_pred = generate_smooth_curve_data(qualities, prices, reqscore)

        # Create chart data in JSON format
    chart_data = {
        "labels": [str(q) for q in qualities],  # Convert qualities to strings for labels
        "prices": prices,
        "predicted_prices": list(y_smooth_pred),  # Convert numpy array to list
        "predicted_qualities": list(X_smooth),  # Add predicted qualities
        "reqscore": reqscore,
        "dates": dates,
        "comments": comments,
        "predicted_price": predicted_price,
        "upper_bound": upper_bound,
        "actual_price": actual_price,
        "search_width": search_width
    }

    # Make overall status "Completed" if no error messages were set
    status_message = "Completed"
    info_messages_list = []  # Use a list to build info message parts
    # Add info about points deleted
    if percentile_message:
        info_messages_list.append(f"{percentile_message}")
    if deleted_count > 0:
        info_messages_list.append(f"{deleted_count} points deleted")
    # Add info about data added
    if add_data == "True" and discogs_data:
        info_messages_list.append(f"Data added to previous run")

    # Join info messages with a newline if there are any
    if info_messages_list:
        info_message = "\n".join(info_messages_list)

    error_messages_list = []
    # send an error message too if there are less than 10 points
    if len(processed_grid)<10:
        error_messages_list.append(f"Less than 10 data points. Add more data if possible")

    # send an error
    if upper_bound<predicted_price:
        error_messages_list.append(f"Max price calc error")

    # Join info messages with a newline if there are any
    if error_messages_list:
        error_message = "\n".join(error_messages_list)

    # output for sending to flask
    output_data = {
        "calculated_price": round(predicted_price, 2),
        "upper_bound": round(upper_bound, 2),
        "actual_price": round(actual_price, 2),
        "status_message": status_message,
        "info_message": info_message,
        "error_message": error_message,
        "chart_data": chart_data  # Include chart data in the output
    }

    return output_data