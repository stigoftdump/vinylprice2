# /home/stigoftdump/PycharmProjects/PythonProject/vinylprice/vin.py
from functions import predict_price, get_actual_price, generate_smooth_curve_data, fit_curve_and_get_params
from grid_functions import extract_tuples, manage_processed_grid


# --- MODIFIED function signature ---
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

    # --- For now, just print the received ID to confirm it's being passed ---
    if discogs_release_id:
        print(f"VIN.PY: Received Discogs Release ID: {discogs_release_id}")
    else:
        print("VIN.PY: No Discogs Release ID received.")
    # --- This print can be removed once API integration starts ---

    output_data = {
        "calculated_price": None,
        "upper_bound": None,
        "actual_price": None,
        "status_message": None,  # Will be set to "Completed" or an error later
        "info_message": None,
        "error_message": None,
        "chart_data": {}
    }

    try:
        # if discogs data is empty, just load the saves processed_grid and use that, otherwise do the whole thing.
        processed_grid, deleted_count, status_from_parsing, api_data = manage_processed_grid(
            discogs_data,
            start_date,
            points_to_delete_json,
            add_data,
            discogs_release_id  # --- NEW: Pass ID to manage_processed_grid ---
        )

        # Store API data in output_data if available
        if api_data:
            output_data["api_artist"] = api_data.get("api_artist")
            output_data["api_title"] = api_data.get("api_title")
            output_data["api_year"] = api_data.get("api_year")
            output_data["api_original_year"] = api_data.get("api_original_year")
            # You can add more API fields here if needed later
            if api_data.get("api_artist") or api_data.get("api_title"):
                 print(f"VIN.PY: API Data processed: {api_data.get('api_artist', '')} - {api_data.get('api_title', '')}")

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

        # If all calculations are successful, populate output_data
        output_data["calculated_price"] = round(predicted_price, 2)
        output_data["upper_bound"] = round(upper_bound, 2)
        output_data["actual_price"] = round(actual_price, 2)
        output_data["chart_data"] = chart_data

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
