from functions import graph_logic
from grid_functions import make_processed_grid, load_processed_grid, save_processed_grid, delete_points

def calculate_vin_data(reqscore, shop_var, start_date, add_data, discogs_data, points_to_delete_json):
    status_message = None
    deleted_count = 0

    # Gets the processed_grid from the discogs_data sent over
    processed_grid, status_message = make_processed_grid(discogs_data, start_date)

    if status_message is not None and not processed_grid: # Check if error occurred *and* grid is empty
        # Output JSON error message from make_processed_grid
        output_data = {"calculated_price": None, "upper_bound": None, "actual_price": None, "status_message": status_message, "chart_data": {}}
        return output_data

    # deletes points if needed
    processed_grid, deleted_count = delete_points(points_to_delete_json, processed_grid)

    # If add_data is True, load the previously saved processed_grid and add it to the current one
    if add_data == "True":
        saved_processed_grid = load_processed_grid()
        processed_grid.extend(saved_processed_grid) # Add saved data *after* potential deletion

    # Check again if the grid is empty after adding saved data or max price filtering
    if not processed_grid:
         status_message = "No data points available for analysis."
         output_data = {"calculated_price": None, "upper_bound": None, "actual_price": None, "status_message": status_message, "chart_data": {}}
         return output_data

    # gets dates and comments from the PROCESSED_GRID to go in the output
    dates = []
    comments = []
    # Ensure the loop handles the structure correctly, especially after filtering
    for row in processed_grid:
        # Check indices carefully
        if len(row) > 0:
            dates.append(row[0]) # Date is index 0
        else:
            dates.append(None) # Or handle error

        if len(row) > 6:
            comments.append(row[6]) # Comment is index 6
        else:
            comments.append("") # Default comment

    # Save processed grid to file
    save_processed_grid(processed_grid)

    # graph logic to get the variables for the output
    # Make sure graph_logic handles potentially empty lists gracefully if grid becomes empty
    try:
        qualities, prices, X_smooth, y_smooth_pred, predicted_price, upper_bound, actual_price = graph_logic(reqscore, shop_var, processed_grid)
    except Exception as e:
        # Handle potential errors in graph_logic if the grid is unusual after filtering
        status_message = f"Error during graph calculation: {e}"
        output_data = {"calculated_price": None, "upper_bound": None, "actual_price": None, "status_message": status_message, "chart_data": {}}
        return output_data

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
        "actual_price": actual_price
    }

    # make status "Okay" if there are no problems
    if status_message is None:
        status_message = "Completed"
        if deleted_count > 0:
            status_message += f". {deleted_count} points deleted"

    # output for sending to flask
    output_data = {
        "calculated_price": round(predicted_price, 2),
        "upper_bound": round(upper_bound, 2),
        "actual_price": round(actual_price, 2),
        "status_message": status_message,
        "chart_data": chart_data  # Include chart data in the output
    }

    return output_data