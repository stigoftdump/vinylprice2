from functions import return_variables, graph_logic
from grid_functions import make_processed_grid, load_processed_grid, save_processed_grid, delete_above_max_price
import sys
import json
import math

# --- ADD HELPER FUNCTION for robust comparison ---
def points_match(grid_row, point_to_delete, tolerance=0.001):
    """
    Checks if a grid row matches a point to be deleted.
    Handles float comparisons with tolerance and missing comments.
    Assumes grid_row structure: (date, quality1, quality2, price, ..., score, optional_comment)
    Assumes point_to_delete structure: {'quality': float, 'price': float, 'date': str, 'comment': str or None}
    """
    if len(grid_row) < 6: # Needs at least date, price, score
        return False

    grid_score = grid_row[5]
    grid_price = grid_row[3]
    grid_date = grid_row[0]
    grid_comment = grid_row[6] if len(grid_row) > 6 else "" # Handle potential missing comment in grid

    # Use get with defaults for the dictionary
    delete_score = point_to_delete.get('quality')
    delete_price = point_to_delete.get('price')
    delete_date = point_to_delete.get('date')
    delete_comment = point_to_delete.get('comment', "") # Default to empty string if missing

    # Check types and existence before comparison
    if delete_score is None or delete_price is None or delete_date is None:
        return False # Cannot match if essential data is missing in point_to_delete

    # Perform comparisons
    # Use math.isclose for float comparison
    score_match = math.isclose(grid_score, delete_score, rel_tol=tolerance)
    price_match = math.isclose(grid_price, delete_price, rel_tol=tolerance)
    date_match = (grid_date == delete_date)
    # Treat None/empty string comments as the same
    comment_match = ( (grid_comment or "") == (delete_comment or "") )

    return score_match and price_match and date_match and comment_match
# --- END ADD HELPER FUNCTION ---

def main():
    status_message = None

    # gets the inputs from the sys arguments
    reqscore, shop_var, start_date, add_data, max_price, discogs_data, points_to_delete_json = return_variables(sys.argv)

    # Ensure max_price is either an integer or None
    if max_price and max_price.isdigit():  # Check if it's not empty and is a number
        max_price = int(max_price)
    else:
        max_price = None  # Set to None explicitly if it's empty or invalid

    # Gets the processed_grid from the discogs_data sent over
    processed_grid, status_message = make_processed_grid(discogs_data, start_date)

    if status_message is not None and not processed_grid: # Check if error occurred *and* grid is empty
        # Output JSON error message from make_processed_grid
        output_data = {"calculated_price": None, "upper_bound": None, "actual_price": None, "status_message": status_message, "chart_data": {}}
        print(json.dumps(output_data))
        sys.exit(0)

    # --- ADD Deletion Logic ---
    points_to_delete = []
    if points_to_delete_json:
        try:
            points_to_delete = json.loads(points_to_delete_json)
        except json.JSONDecodeError:
            # Log error or set a status message if desired, but continue with empty list
            print("Warning: Could not decode points_to_delete JSON. Proceeding without deleting points.", file=sys.stderr)
            points_to_delete = [] # Ensure it's an empty list

    if points_to_delete and processed_grid: # Only filter if there are points to delete and a grid exists
        initial_count = len(processed_grid)
        filtered_grid = []
        for row in processed_grid:
            should_delete = False
            for point in points_to_delete:
                if points_match(row, point):
                    should_delete = True
                    break # Found a match, no need to check other points_to_delete for this row
            if not should_delete:
                filtered_grid.append(row)

        deleted_count = initial_count - len(filtered_grid)
        if deleted_count > 0:
             print(f"Info: Deleted {deleted_count} point(s) based on selection.", file=sys.stderr) # Optional info message

        processed_grid = filtered_grid # Replace the grid with the filtered version
    # --- END Deletion Logic ---

    # adds in the maxprice and deletes if anything is above it.
    if max_price is not None:
        # Make sure delete_above_max_price can handle an empty grid gracefully
        if processed_grid:
            processed_grid=delete_above_max_price(processed_grid,max_price)

    # If add_data is True, load the previously saved processed_grid and add it to the current one
    if add_data == "True":
        saved_processed_grid = load_processed_grid()
        processed_grid.extend(saved_processed_grid) # Add saved data *after* potential deletion

    # Check again if the grid is empty after adding saved data or max price filtering
    if not processed_grid:
         status_message = "No data points available for analysis."
         output_data = {"calculated_price": None, "upper_bound": None, "actual_price": None, "status_message": status_message, "chart_data": {}}
         print(json.dumps(output_data))
         sys.exit(0)

    # gets dates and comments from the PROCESSED_GRID to go in the json
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
        qualities, prices, X_smooth, y_smooth_pred, reqscore_out, predicted_price, upper_bound, actual_price = graph_logic(reqscore, shop_var, processed_grid)
    except Exception as e:
        # Handle potential errors in graph_logic if the grid is unusual after filtering
        status_message = f"Error during graph calculation: {e}"
        output_data = {"calculated_price": None, "upper_bound": None, "actual_price": None, "status_message": status_message, "chart_data": {}}
        print(json.dumps(output_data))
        sys.exit(0)


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

    # output for sending to flask
    output_data = {
        "calculated_price": round(predicted_price, 2),
        "upper_bound": round(upper_bound, 2),
        "actual_price": round(actual_price, 2),
        "status_message": status_message,
        "chart_data": chart_data  # Include chart data in the output
    }

    print(json.dumps(output_data))  # Print the JSON string

if __name__ == "__main__":
    main()