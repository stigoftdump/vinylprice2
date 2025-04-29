import pickle
import re
from datetime import datetime
import math
import json
import sys

# Mapping of quality to numeric value
quality_to_number = {
    'Mint (M) ': 9,
    'Near Mint (NM or M-) ': 8,
    'Very Good Plus (VG+) ': 7,
    'Very Good (VG) ': 6,
    'Good Plus (G+) ': 5,
    'Good (G) ': 4,
    'Fair (F) ': 3,
    'Poor (P) ': 2,
    'Generic ': 1,
    'Not Graded ': 1,
    'No Cover ': 1
}

# list of real prices
real_prices =[
    1.99,
    2.99,
    3.99,
    4.99,
    5.99,
    6.99,
    7.99,
    8.99,
    9.99,
    12.99,
    14.99,
    17.99,
    19.99,
    22.99,
    24.99,
    27.99,
    29.99,
    34.99,
    39.99
]

# function to save processed grid to a file
def save_processed_grid(processed_grid, filename='processed_grid.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(processed_grid, f)

# function to load a saved processed grid from a file
def load_processed_grid(filename='processed_grid.pkl'):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []  # Return an empty list if the file doesn't exist

# converts the record quality and sleeve quality into a score
def calculate_score(record_quality, sleeve_quality):
    record_value = quality_to_number.get(record_quality, 0)
    sleeve_value = quality_to_number.get(sleeve_quality, 0)
    score = record_value - ((record_value - sleeve_value) / 3)
    return score

# gets the sale price from the calculated price
def realprice(pred_price):
    if pred_price <42.48:
        # Finds the closest price using the min function with a custom key
        foundprice= min(real_prices, key=lambda x: abs(x - pred_price))
    elif pred_price<100:
        nearest_divisible_by_5 = round(pred_price / 5) * 5
        foundprice = nearest_divisible_by_5
    else:
        nearest_divisible_by_10 = round(pred_price / 10) * 10
        foundprice = nearest_divisible_by_10
    return foundprice

# creates the processed grid data from imported data
def make_processed_grid(clipboard_content, start_date):

    # status_message is blank for now
    status_message = None
    processed_grid = None

    # Check for the presence of "Order Date" and "Change Currency" in the clipboard content
    if "Order Date" not in clipboard_content or "Change Currency" not in clipboard_content:
        # return status that no data is there
        return None, "No Discogs Data in text box"
    else:
        # Split content into rows based on newlines
        rows = clipboard_content.splitlines()  # 'rows' is defined here

        # Extract the portion of the clipboard content starting from "Order Date" and stopping before "Change Currency"
        start_index = None
        end_index = None

        for i, row in enumerate(rows):
            if "Order Date" in row:
                start_index = i
            if "Change Currency" in row and start_index is not None:
                end_index = i
                break

        # Ensure valid indices are found
        if start_index is not None and end_index is not None:
            rows = rows[start_index:end_index]

        # Get the relevant rows (excluding header, up to before "Change Currency")
        relevant_rows = rows[start_index + 1: end_index]

        # --- MODIFICATION START ---

        intermediate_grid = []  # Build the grid row by row here
        try:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
        except ValueError:
            return None, "Invalid start_date format. Please use YYYY-MM-DD."

        # Compile regex for matching the date at the start of a line
        # Allows for potential whitespace before the date
        date_pattern = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})")

        i = 0
        while i < len(relevant_rows):
            row = relevant_rows[i]
            match = date_pattern.match(row)

            # Check if the current line is a data line (starts with a date)
            if match:
                try:
                    # Split the data row by tabs
                    data_parts = row.split('\t')

                    # --- MODIFIED: Price handling ---
                    original_price_native = None  # To track if we split price (likely Windows format)
                    price_in_user_currency_str = None

                    if len(data_parts) > 3:
                        price_cell_content = data_parts[3]
                        if isinstance(price_cell_content, str):
                            if '\n' in price_cell_content:
                                prices = price_cell_content.split('\n', 1)
                                price_in_user_currency_str = prices[0]
                                if len(prices) > 1:
                                    original_price_native = prices[1]  # Indicates potential Windows format
                            else:
                                price_in_user_currency_str = price_cell_content  # Linux format

                    # Ensure enough parts for basic processing (Date, Q1, Q2, Price)
                    # We need at least 4 parts based on the split of the *first* line
                    if len(data_parts) < 4 or price_in_user_currency_str is None:
                        print(f"Warning: Skipping malformed data row (missing columns or price): {row}")
                        i += 1
                        continue  # Skip to next line

                    # Convert tuple to list for modification
                    current_row_list = list(data_parts)  # Note: data_parts[3] might still contain the split string

                    # --- Price Conversion (using extracted string) ---
                    price_float = None
                    if price_in_user_currency_str and price_in_user_currency_str.startswith('£'):
                        try:
                            clean_price = price_in_user_currency_str[1:].replace(',', '')
                            price_float = float(clean_price)
                        except ValueError:
                            print(
                                f"Warning: Could not convert price '{price_in_user_currency_str}' to number in row: {row}")
                            # Keep price_float as None
                    elif price_in_user_currency_str:
                        # Attempt conversion even if no '£', useful if currency changes
                        try:
                            price_float = float(price_in_user_currency_str.replace(',', ''))
                        except ValueError:
                            print(
                                f"Warning: Could not convert non-£ price '{price_in_user_currency_str}' to number in row: {row}")
                            # Keep price_float as None

                    # Replace the original price cell (or split string) with the float or None
                    current_row_list[3] = price_float

                    # --- Score Calculation ---
                    score = None
                    if len(current_row_list) > 2:
                        record_quality = current_row_list[1]
                        sleeve_quality = current_row_list[2]
                        # Add strip() here in case qualities have leading/trailing spaces
                        score = calculate_score(record_quality.strip(), sleeve_quality.strip())
                    # Append score (even if None) - ensure it's always the 6th element (index 5) if price is index 3
                    # Need to handle cases where original data had more/fewer tabs carefully.
                    # Let's assume the structure is consistent enough for now and append score.
                    # If the original data had more than 4 tabs, this needs adjustment.
                    # For safety, let's ensure the list has enough elements first.
                    while len(current_row_list) < 5:
                        current_row_list.append(None)  # Pad if needed before score
                    current_row_list = current_row_list[:5]  # Trim excess if > 5 before score
                    current_row_list.append(score)  # Score becomes index 5

                    # --- REVISED: Check for Comment on NEXT line(s) ---
                    comment = None
                    lines_to_skip_ahead = 0  # How many EXTRA lines to advance 'i' by

                    # Check line i+1
                    if i + 1 < len(relevant_rows):
                        next_row_1 = relevant_rows[i + 1]
                        # Scenario 1: Comment is on the next line (Linux style)
                        if next_row_1.strip().startswith('Comments:'):
                            comment_text = next_row_1.strip()
                            comment = comment_text[len("Comments:"):].strip()
                            lines_to_skip_ahead = 1  # Skip this comment line
                        # Scenario 2: Check line i+2 if line i+1 was likely native price (Windows style)
                        elif original_price_native is not None and i + 2 < len(relevant_rows):
                            next_row_2 = relevant_rows[i + 2]
                            if next_row_2.strip().startswith('Comments:'):
                                comment_text = next_row_2.strip()
                                comment = comment_text[len("Comments:"):].strip()
                                lines_to_skip_ahead = 2  # Skip native price line AND comment line

                    if comment is not None:
                        # Append comment - becomes 7th element (index 6)
                        current_row_list.append(comment)

                    # Add the fully processed row
                    intermediate_grid.append(current_row_list)

                    # Advance 'i' past the current line PLUS any consumed comment/price lines
                    i += (1 + lines_to_skip_ahead)
                    continue  # Continue to next iteration of while loop

                except ValueError as ve:
                    # Handle potential errors during date parsing or float conversion within the loop
                    print(f"Warning: Skipping row due to data conversion error: {row} | Error: {ve}")
                except IndexError as ie:
                    # Handle potential errors if a row has fewer columns than expected
                    print(f"Warning: Skipping row due to missing columns: {row} | Error: {ie}")

            # If the line didn't start with a date, or after handling exceptions/skips,
            # ensure we always advance 'i' if it wasn't advanced by 'continue' above.
            # This check handles lines that are not data lines (like the native price or comments
            # that weren't consumed, or blank lines).
            if not match or 'lines_to_skip_ahead' not in locals() or lines_to_skip_ahead == 0:
                i += 1
                # Reset lines_to_skip_ahead if it exists from a previous iteration that failed
                # This part might be redundant if the continue statement handles all successful cases
                if 'lines_to_skip_ahead' in locals():
                    del lines_to_skip_ahead

        # Convert the list of lists back to a list of tuples for the final output
        processed_grid_final = [tuple(item) for item in intermediate_grid]

    return processed_grid_final, status_message

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

def delete_points(points_to_delete_json, processed_grid):
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

    return processed_grid