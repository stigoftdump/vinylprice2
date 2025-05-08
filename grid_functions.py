import re
from datetime import datetime
import math
import json
import sys

# Mapping of quality to numeric value - NO TRAILING SPACES IN KEYS
quality_to_number = {
    'Mint (M)': 9,
    'Near Mint (NM or M-)': 8,
    'Very Good Plus (VG+)': 7,
    'Very Good (VG)': 6,
    'Good Plus (G+)': 5,
    'Good (G)': 4,
    'Fair (F)': 3,
    'Poor (P)': 2,
    'Generic': 1,
    'Not Graded': 1,
    'No Cover': 1
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

# Inflation rates (Year: Rate as decimal) - Assuming rates are for the year ending in that year
# These are based off the RPI for "Entertainment and other recreations", the category which includes vinyl
inflation_rates = {
    2020: 0.02,
    2021: 0.025,
    2022: 0.061,
    2023: 0.044,
    2024: 0.06,
}

def adjust_for_inflation(price, sale_year):
    """
    Adjusts a historical price to its equivalent value in a target year using inflation rates.

    Args:
        price (float): The original price from the sale year.
        sale_year (int): The year the sale occurred.
        target_inflation_year (int, optional): The year to adjust the price to. Defaults to 2024.

    Returns:
        float: The inflation-adjusted price, rounded to 2 decimal places.
               Returns the original price if adjustment cannot be fully completed
               due to missing inflation rates.
    """
    target_inflation_year = 2024  # Adjust all prices to 2025 values
    adjustment_factor = 1.0

    # Iterate through years from sale_year up to the target year
    for year in range(sale_year, target_inflation_year + 1):
        if year in inflation_rates:
            adjustment_factor *= (1 + inflation_rates[year])
        else:
            # If inflation rate for a year is missing, we cannot adjust accurately
            print(f"Warning: Missing inflation rate for year {year}. Inflation adjustment stopped at previous year.", file=sys.stderr)
            break # Stop adjustment if rate is missing

    new_price = round(price * adjustment_factor,2)

    return new_price

# converts the record quality and sleeve quality into a score
def calculate_score(record_quality, sleeve_quality):
    """
    Calculates a combined quality score based on media and sleeve grades.

    The score prioritizes media quality, with sleeve quality acting as a modifier.
    Uses the `quality_to_number` mapping to convert grades to numeric values.

    Args:
        record_quality (str): The quality grade string for the media (e.g., 'Near Mint (NM or M-)').
        sleeve_quality (str): The quality grade string for the sleeve.

    Returns:
        float: The calculated quality score (typically between 1 and 9).
               Returns 0.0 if the record quality grade is unrecognized.
    """
    # Strip input strings to match the keys (which now lack trailing spaces)
    record_value = quality_to_number.get(record_quality.strip(), 0)
    sleeve_value = quality_to_number.get(sleeve_quality.strip(), 0)


    if record_value == 0: # If record quality is unknown/unmapped, score is 0
        return 0.0
    # Calculate score even if sleeve quality is unknown (treats sleeve_value as 0)
    score = record_value - ((record_value - sleeve_value) / 3)
    return score

# gets the sale price from the calculated price
def realprice(pred_price):
    """
    Rounds a predicted price to the nearest 'realistic' price point.

    Uses a predefined list (`real_prices`) for lower values, rounds to the
    nearest 5 for mid-range values, and nearest 10 for higher values.

    Args:
        pred_price (float): The predicted price calculated by the model.

    Returns:
        float: The rounded, 'realistic' price.
    """
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
    """
    Parses raw text data (presumably pasted from Discogs sales history)
    into a structured grid format.

    Extracts date, media/sleeve condition, price, native price, comments,
    calculates a quality score, and adjusts price for inflation. Filters
    entries based on the start date.

    Args:
        clipboard_content (str): The raw text data pasted by the user.
        start_date_str (str): The earliest date ('YYYY-MM-DD') for sales data to include.

    Returns:
        tuple: A tuple containing:
            - processed_grid (list): A list of tuples, each representing a processed sale.
                                     Format: (date_str, media_grade, sleeve_grade,
                                             adjusted_price_float, native_price_str,
                                             score_float, comment_str)
            - status_message (str or None): An error message if parsing fails, otherwise None.
    """
    # Initialize status message to None (no error initially)
    status_message = None
    # Initialize an empty list to store the processed rows
    processed_grid = []

    # Basic validation: Check if essential keywords are present in the input text
    if "Order Date" not in clipboard_content or "Change Currency" not in clipboard_content:
        # If keywords are missing, assume it's not valid Discogs data
        return [], "No Discogs Data in text box"

    # Validate and parse the start_date string
    try:
        # Convert the start_date string ('YYYY-MM-DD') into a date object
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
    except ValueError:
        # If the date format is invalid, return an error message
        return [], "Invalid start_date format. Please use YYYY-MM-DD."

    # Split the input text into individual lines (rows)
    all_rows = clipboard_content.splitlines()
    # Initialize index for the start of relevant data (-1 means not found yet)
    start_index = -1
    # Initialize index for the end of relevant data (default to end of list)
    end_index = len(all_rows)

    # Iterate through the rows to find the header and footer markers
    for i, row_text in enumerate(all_rows):
        # Look for the header row containing specific column names
        if "Order Date" in row_text and "Condition" in row_text and "Sleeve Condition" in row_text and start_index == -1:
            # Set start_index to the line *after* the header
            start_index = i + 1
        # Look for the footer marker "Change Currency" after the header has been found
        elif "Change Currency" in row_text and start_index != -1:
            # Set end_index to the line *before* the footer
            end_index = i
            # Stop searching once the footer is found
            break

    # If the full header wasn't found, try a simpler check just for "Order Date"
    if start_index == -1:
        for i, row_text in enumerate(all_rows):
             if "Order Date" in row_text and start_index == -1:
                 start_index = i + 1
                 break
        # If even "Order Date" wasn't found, return an error
        if start_index == -1:
             return [], "Could not find 'Order Date' header row."

    # Extract the rows between the header and footer
    relevant_rows = all_rows[start_index:end_index]
    # Compile a regular expression to match dates at the beginning of a line (YYYY-MM-DD)
    date_pattern = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})")

    # Initialize loop counter for iterating through relevant rows
    i = 0
    # Loop through the relevant rows
    while i < len(relevant_rows):
        # Get the current row
        row = relevant_rows[i]
        # Try to match the date pattern at the start of the row
        match = date_pattern.match(row)

        # If a date is found at the beginning of the row
        if match:
            # Initialize variables for the extracted data for this sale entry
            date_str_out = None
            quality1_str_raw = None # Raw media quality string from input
            quality2_str_raw = None # Raw sleeve quality string from input
            price_float_out = None  # Inflation-adjusted price
            native_price_str_out = None # Price in original currency
            score_out = None        # Calculated quality score
            comment_str_out = None  # Sale comment
            lines_to_skip_ahead = 0 # How many extra lines to skip (for multi-line entries)

            # Use a try-except block to handle potential errors during row processing
            try:
                # Split the row into parts based on tab characters
                data_parts = row.split('\t')
                # Check if there are enough parts (at least date, media, sleeve, price)
                if len(data_parts) < 4:
                    # If not enough parts, print a warning and skip the row
                    print(f"Warning: Skipping row, expected >= 4 tab parts: {row}")
                    i += 1
                    continue # Move to the next iteration of the while loop

                # --- Extract Date ---
                # Get the date string from the first part and remove leading/trailing whitespace
                date_str_raw = data_parts[0].strip()
                # Convert the date string to a date object
                date_obj = datetime.strptime(date_str_raw, '%Y-%m-%d').date()

                # --- Filter by Date ---
                # Check if the sale date is before the specified start date
                if date_obj < start_date_obj:
                    # If too early, skip this row and move to the next
                    i += 1
                    continue

                # Format the date string for output (add a trailing space, though this might be unnecessary)
                date_str_out = date_str_raw # Removed the trailing space addition + ' '

                # --- Extract Quality Grades ---
                # Get the raw media quality string (might have trailing spaces)
                quality1_str_raw = data_parts[1]
                # Get the raw sleeve quality string (might have trailing spaces)
                quality2_str_raw = data_parts[2]

                # --- Extract and Process Price ---
                # Get the price string (potentially split by newline on Windows)
                price_user_str = data_parts[3]
                # If the price string contains a newline, take only the part before it
                if '\n' in price_user_str:
                    price_user_str = price_user_str.split('\n')[0]

                # Check if a price string was actually extracted
                if price_user_str:
                    try:
                        # Remove currency symbols (£$€) and commas from the price string
                        cleaned_price = re.sub(r'[£$€,]','', price_user_str)
                        # Convert the cleaned price string to a float
                        original_price = float(cleaned_price)
                        # Get the year from the sale date object
                        sale_year = date_obj.year
                        # Adjust the original price for inflation based on the sale year
                        price_float_out = adjust_for_inflation(original_price, sale_year)
                    except ValueError:
                        # Handle error if price conversion fails
                        print(f"Warning: Could not convert user price '{price_user_str}' to float.")
                        price_float_out = None
                    except Exception as e:
                        # Handle error during inflation adjustment
                        print(f"Warning: Error during inflation adjustment for price '{price_user_str}' (year {date_obj.year}): {e}", file=sys.stderr)
                        price_float_out = None # Set price to None if adjustment fails
                else:
                     # If no price string was found, set the output price to None
                     price_float_out = None

                # --- Determine Format and Extract Native Price / Comment ---
                # Check if the row has 5 or more parts (typical Linux copy-paste format)
                if len(data_parts) >= 5: # Linux Format
                    # The native price is in the 5th part
                    native_price_str_out = data_parts[4].strip()
                    # Check if the *next* line exists
                    if i + 1 < len(relevant_rows):
                        # Get the next line
                        next_row_1 = relevant_rows[i + 1]
                        # Check if the next line starts with "Comments:"
                        if next_row_1.strip().startswith('Comments:'):
                            # Extract the comment text after "Comments:"
                            comment_text = next_row_1.strip()
                            comment_str_out = comment_text[len("Comments:"):].strip()
                            # We need to skip this comment line in the next iteration
                            lines_to_skip_ahead = 1
                # Check if the row has exactly 4 parts (typical Windows copy-paste format)
                elif len(data_parts) == 4: # Windows Format
                    # Check if the *next* line exists (this should contain the native price)
                    if i + 1 < len(relevant_rows):
                        # Extract the native price from the next line
                        native_price_str_out = relevant_rows[i + 1].strip()
                        # We need to skip this native price line in the next iteration
                        lines_to_skip_ahead = 1
                        # Check if the line *after* the native price exists
                        if i + 2 < len(relevant_rows):
                            # Get the line after the native price
                            next_row_2 = relevant_rows[i + 2]
                            # Check if this line starts with "Comments:"
                            if next_row_2.strip().startswith('Comments:'):
                                # Extract the comment text
                                comment_text = next_row_2.strip()
                                comment_str_out = comment_text[len("Comments:"):].strip()
                                # We need to skip both the native price and comment lines
                                lines_to_skip_ahead = 2
                    else:
                        # Handle case where Windows format is expected but next line is missing
                        print(f"Warning: Windows format, line i+1 missing: {row}")

                # --- Calculate Score ---
                # Calculate the quality score using the raw quality strings
                # (calculate_score function handles stripping whitespace)
                score_out = calculate_score(quality1_str_raw, quality2_str_raw)

                # --- Assemble Output Tuple ---
                # Create a tuple containing all the processed data for this sale
                # Note: Quality strings are stripped *before* adding to the tuple for consistency
                processed_row_tuple = (
                    date_str_out,
                    quality1_str_raw.strip(), # Use stripped string for output
                    quality2_str_raw.strip(), # Use stripped string for output
                    price_float_out,
                    native_price_str_out,
                    score_out,
                    comment_str_out
                )
                # Add the processed tuple to the results list
                processed_grid.append(processed_row_tuple)

                # Increment the loop counter by 1 (for the current row) plus any extra lines skipped
                i += (1 + lines_to_skip_ahead)

            # --- Error Handling for Row Processing ---
            except Exception as e:
                # Print an error message if any unexpected error occurs while processing a row
                print(f"Error processing data row starting with '{row[:50]}...': {e}")
                # Increment the counter to move to the next line even if an error occurred
                i += 1
        # If the row does not start with a date, simply skip it
        else:
            i += 1

    # Return the list of processed sale tuples and the status message (which is likely still None)
    return processed_grid, status_message

def points_match(grid_row, point_to_delete, tolerance=0.001):
    """
    Checks if a row from the processed grid matches a point selected for deletion.

    Compares date, quality score, price, and comment, handling potential None values
    and using tolerance for float comparisons.

    Args:
        grid_row_tuple (tuple): A tuple representing a row from the processed_grid.
                                Expected format matches make_processed_grid output.
        point_to_delete_dict (dict): A dictionary representing a point selected in the UI.
                                     Expected keys: 'quality', 'price', 'date', 'comment'.
        tolerance (float, optional): Tolerance for floating-point comparisons (score, price).
                                     Defaults to 0.001.

    Returns:
        bool: True if the row matches the point to delete, False otherwise.
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
    """
    Filters the processed grid, removing rows that match points marked for deletion.

    Args:
        points_to_delete_json (str): A JSON string representing a list of points
                                     (dictionaries) selected for deletion in the UI.
        processed_grid (list): The list of processed sale data tuples.

    Returns:
        tuple: A tuple containing:
            - filtered_grid (list): The processed grid with matched points removed.
            - deleted_count (int): The number of points removed from the grid.
    """
    points_to_delete = []
    if points_to_delete_json:
        try:
            points_to_delete = json.loads(points_to_delete_json)
        except json.JSONDecodeError:
            # Log error or set a status message if desired, but continue with empty list
            print("Warning: Could not decode points_to_delete JSON. Proceeding without deleting points.", file=sys.stderr)
            points_to_delete = [] # Ensure it's an empty list

    deleted_count = 0 # Initialize deleted_count before the conditional block

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

    # Modified line: Return deleted_count as well
    return processed_grid, deleted_count