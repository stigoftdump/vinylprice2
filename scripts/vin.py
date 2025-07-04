from scripts.chart.functions import predict_price, get_actual_price, generate_smooth_curve_data, fit_curve_and_get_params, write_output
from scripts.grid.grid_functions import extract_tuples, manage_processed_grid
from scripts.grid.point_deletion import delete_ml_sales_for_recalled_release
from scripts.grid.api_import import fetch_api_data, fake_api_data
from scripts.persistence import remember_last_run, machine_learning_save
from scripts.grid.processing_grid import execute_data_processing_flow

def calculate_vin_data(reqscore, shop_var, start_date, add_data, discogs_data, points_to_delete_json,
                       discogs_release_id=None):
    """
    Main calculation engine.

    Orchestrates the process of parsing input data, optionally merging with saved data,
    deleting selected points, performing graph fitting and price prediction,
    and preparing the output data including chart information.
    It can also accept a Discogs release ID for future API integration.

    Args:
        reqscore (float): The target quality score for price prediction.
        shop_var (float): The shop variance factor to apply to the price uncertainty.
        start_date (str): The start date ('YYYY-MM-DD') for filtering sales data.
        add_data (str): String ('True' or 'False') indicating whether to merge
                            current data with previously saved data.
        discogs_data (str): Raw text data pasted by the user (Discogs sales history).
        points_to_delete_json (str): JSON string array of points selected for deletion.
        discogs_release_id (str, optional): The Discogs release ID. Defaults to None.

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
    output_data = {}

    try:
        # Handles the grid processing and api_ddata
        processed_grid, api_data, status_messages, deleted_count = execute_data_processing_flow(discogs_data, discogs_release_id,points_to_delete_json,add_data, start_date)

        # Extracts elements
        qualities, prices, dates, comments = extract_tuples(processed_grid)

        # Get the fitted parameters and prediction function
        params, predict_func = fit_curve_and_get_params(qualities, prices)

        # gets the predicted price
        predicted_price = predict_price(reqscore, predict_func)

        # gets the actual price from the predicted price
        upper_bound, actual_price, search_width = get_actual_price(reqscore, shop_var, qualities, prices,
                                                                   predicted_price, predict_func)

        # Gets the smoothed data for the chart
        X_smooth, y_smooth_pred = generate_smooth_curve_data(qualities, prices, reqscore, predict_func)

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

        # writes the output data
        output_data = write_output(api_data, predicted_price, upper_bound, actual_price, chart_data)

        # Make overall status "Completed" if no error messages were set
        output_data["status_message"] = "Completed"
        output_data["status_message"] = "Completed"

        # Build status and info messages if successful
        info_messages_list = []  # Use a list to build info message parts
        # Add info about points deleted
        if deleted_count > 0:
            info_messages_list.append(f"{deleted_count} points deleted")
        # Add info about data added
        if add_data == "True" and discogs_data:
            info_messages_list.append(f"Data added to previous run")

        # puts all the info messages together
        if info_messages_list:
            output_data["info_message"] = "\n".join(info_messages_list)

        # Additional error checks
        error_messages_list = []
        if len(processed_grid) < 10:
            error_messages_list.append(f"Less than 10 data points. Add more data if possible")
        if upper_bound < predicted_price:
            error_messages_list.append(f"Max price calc error")
        if error_messages_list:
            output_data["error_message"] = "\n".join(error_messages_list)

    except ValueError as e:
        output_data["error_message"] = f"Validation Error: {e}"
        output_data["status_message"] = "Failed"
    except TypeError as e:
        output_data["error_message"] = f"Type Error: {e}"
        output_data["status_message"] = "Failed"
    except Exception as e:
        # Catch-all for any other unexpected errors
        output_data["error_message"] = f"An unexpected error occurred: {e}"
        output_data["status_message"] = "Failed"

    return output_data
