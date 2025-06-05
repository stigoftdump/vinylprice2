import re
from datetime import datetime
import math
import json
import sys
from persistence import read_save_value, write_save_value, read_ml_data, write_ml_data

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

def get_relevant_rows(clipboard_content):
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

    return relevant_rows

def extract_price(price_string, date_obj):
    # --- Extract and Process Price ---
    # Get the price string (potentially split by newline on Windows)
    price_user_str = price_string
    # If the price string contains a newline, take only the part before it
    if '\n' in price_user_str:
        price_user_str = price_user_str.split('\n')[0]

    # Check if a price string was actually extracted
    if price_user_str:
        try:
            # Remove currency symbols (£$€) and commas from the price string
            cleaned_price = re.sub(r'[£$€,]', '', price_user_str)
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
            print(
                f"Warning: Error during inflation adjustment for price '{price_user_str}' (year {date_obj.year}): {e}",
                file=sys.stderr)
            price_float_out = None  # Set price to None if adjustment fails
    else:
        # If no price string was found, set the output price to None
        price_float_out = None

    return price_float_out

def calculate_line_skip(part_length, iteration, relevant_rows):
    """Determines how many extra lines (beyond the current one) to skip."""
    # Assume no lines to skip ahead unless stated
    lines_to_skip_ahead = 0

    # Checks to see where we are
    if part_length>= 5: # linux
        # are wet the end?
        if iteration + 1 < len(relevant_rows):
            if relevant_rows[iteration + 1].strip().startswith('Comments:'): # does the next line start with "comments:"?
                lines_to_skip_ahead = 1
    elif part_length == 4:# windows
        # the end?
        if iteration + 1 < len(relevant_rows):
            lines_to_skip_ahead = 1
            if iteration + 2 < len(relevant_rows):
                if relevant_rows[iteration + 2].strip().startswith('Comments:'): # does the next line start with "comments:"?
                    lines_to_skip_ahead = 2

    return lines_to_skip_ahead

def extract_native_price(part_length, linux_date, relevant_rows, iteration):
    native_price = None
    # extracts native price
    if part_length >= 5: # linux
        native_price = linux_date.strip()
    elif part_length == 4: # windows
        if iteration+1 < len(relevant_rows):
            native_price = relevant_rows[iteration + 1].strip()
    else:
        native_price = None

    return native_price

def extract_comments(part_length, relevant_rows, iteration):
    # extracts comments
    comment_str_out = ""

    # Check if the row has 5 or more parts (typical Linux copy-paste format)
    if part_length >= 5:  # Linux Format
        # Check if the *next* line exists
        if iteration + 1 < len(relevant_rows):
            # Get the next line
            next_row_1 = relevant_rows[iteration + 1]
            # Check if the next line starts with "Comments:"
            if next_row_1.strip().startswith('Comments:'):
                # Extract the comment text after "Comments:"
                comment_text = next_row_1.strip()
                comment_str_out = comment_text[len("Comments:"):].strip()
    # Check if the row has exactly 4 parts (typical Windows copy-paste format)
    elif part_length == 4:  # Windows Format
        # Check if the *next* line exists (this should contain the native price)
        if iteration + 1 < len(relevant_rows):
            # Check if the line *after* the native price exists
            if iteration + 2 < len(relevant_rows):
                # Get the line after the native price
                next_row_2 = relevant_rows[iteration + 2]
                # Check if this line starts with "Comments:"
                if next_row_2.strip().startswith('Comments:'):
                    # Extract the comment text
                    comment_text = next_row_2.strip()
                    comment_str_out = comment_text[len("Comments:"):].strip()
        else:
            # Handle case where Windows format is expected but next line is missing
            print(f"Warning: Windows format, line i+1 missing")

    return comment_str_out

def _parse_single_entry(relevant_rows, current_index, date_pattern, start_date_obj):
    """
    Attempts to parse a single Discogs sales entry starting from current_index.
    A single entry might span multiple lines due to format differences.

    Args:
        relevant_rows (list): List of strings, each a line from the relevant data block.
        current_index (int): The index in relevant_rows to start parsing from.
        date_pattern (re.Pattern): Compiled regex pattern to match dates.
        start_date_obj (datetime.date): The parsed start date for filtering.

    Returns:
        tuple: (parsed_data_tuple, lines_consumed, error_message_for_entry)
               - parsed_data_tuple (tuple or None): The structured tuple of the parsed entry,
                                                    or None if parsing failed or no entry found.
               - lines_consumed (int): Number of lines from relevant_rows processed for this attempt.
                                       Will be at least 1.
               - error_message_for_entry (str or None): Error message if parsing this specific
                                                        entry failed, otherwise None.
    """
    if current_index >= len(relevant_rows):
        return None, 0, None  # Should not happen if loop is correct, but defensive

    row = relevant_rows[current_index]
    match = date_pattern.match(row)

    if not match:
        return None, 1, None  # No date found at the start of this line, consume 1 line

    try:
        data_parts = row.split('\t')

        # --- Extract Date ---
        date_str_raw = data_parts[0].strip()
        date_obj = datetime.strptime(date_str_raw, '%Y-%m-%d').date()

        # --- Filter by Date ---
        if date_obj < start_date_obj:
            return None, 1, None  # Entry is too old, consume 1 line (the date line)

        date_str_out = date_str_raw

        # --- Extract Quality Grades (assuming at least 3 parts if date matched) ---
        if len(data_parts) < 3:
            raise ValueError(f"Row '{row[:50]}...' started with a date but had less than 3 tab-separated parts.")
        quality1_str_raw = data_parts[1]
        quality2_str_raw = data_parts[2]

        # --- Extract Price (from 4th part if available) ---
        price_string_from_data = data_parts[3] if len(data_parts) > 3 else ""
        price_float_out = extract_price(price_string_from_data, date_obj)

        # --- Determine lines to skip ahead & extract native price/comments ---
        # This uses the existing helper functions
        lines_to_skip_ahead = calculate_line_skip(len(data_parts), current_index, relevant_rows)

        native_price_data_part = data_parts[4].strip() if len(data_parts) > 4 else ""
        native_price_str_out = extract_native_price(len(data_parts), native_price_data_part, relevant_rows,
                                                    current_index)

        comment_str_out = extract_comments(len(data_parts), relevant_rows, current_index)

        # --- Calculate Score ---
        score_out = calculate_score(quality1_str_raw, quality2_str_raw)

        # --- Assemble Output Tuple ---
        processed_row_tuple = (
            date_str_out,
            quality1_str_raw.strip(),
            quality2_str_raw.strip(),
            price_float_out,
            native_price_str_out,
            score_out,
            comment_str_out
        )

        return processed_row_tuple, 1 + lines_to_skip_ahead, None

    except Exception as e:
        error_msg = f"Error processing entry at line {current_index} ('{row[:50]}...'): {e}"
        # On error, we consume at least the current line.
        # If lines_to_skip_ahead was determined before the error, it might be more,
        # but for simplicity and to avoid skipping valid data after an error,
        # consuming just 1 line on error is safer.
        return None, 1, error_msg

def make_processed_grid(clipboard_content, start_date_str_param):  # Renamed param to avoid clash with variable
    """
    Parses raw text data (presumably pasted from Discogs sales history)
    into a structured grid format.
    (Keep existing Args and Returns docstring, but update start_date_str to start_date_str_param if you rename)
    """
    status_message = None  # For overall status, not individual row errors
    processed_grid = []

    # set this to True or False depending on whether you want data saved or not.
    ml_save_setting = True

    if "Order Date" not in clipboard_content or "Change Currency" not in clipboard_content:
        return [], "No Discogs Data in text box"

    try:
        start_date_obj = datetime.strptime(start_date_str_param, '%Y-%m-%d').date()
    except ValueError:
        return [], "Invalid start_date format. Please use YYYY-MM-DD."

    relevant_rows_or_error = get_relevant_rows(clipboard_content)
    if isinstance(relevant_rows_or_error, tuple) and len(
            relevant_rows_or_error) == 2:  # Error case from get_relevant_rows
        return relevant_rows_or_error  # Returns ([], "Error message")

    relevant_rows = relevant_rows_or_error

    date_pattern = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})")
    i = 0
    while i < len(relevant_rows):
        parsed_tuple, lines_consumed, entry_error = _parse_single_entry(
            relevant_rows,
            i,
            date_pattern,
            start_date_obj
        )

        if parsed_tuple:
            processed_grid.append(parsed_tuple)

        if entry_error:  # Log individual entry errors
            print(entry_error, file=sys.stderr)
            # You might want to collect these errors for a more comprehensive status_message

        if lines_consumed == 0:  # Should not happen with current _parse_single_entry logic
            print(
                f"Warning: _parse_single_entry consumed 0 lines at index {i}. Incrementing by 1 to avoid infinite loop.",
                file=sys.stderr)
            i += 1
        else:
            i += lines_consumed

    # save the processed grid for machine learning
    if processed_grid and ml_save_setting is True:
        try:
            # gets metadata
            artist, album, label, extra_comments = extract_record_metadata(clipboard_content)

            machine_learning_save(processed_grid, artist, album, label, extra_comments)
        except Exception as e:
            print(f"Error saving ML data: {e}", file=sys.stderr)

    return processed_grid, status_message

def machine_learning_save(processed_grid, artist, album, label, extra_comments):
    """
    Appends processed sales data points to the ML data file, including extracted metadata,
    while checking for duplicates based on date, artist, album, label,
    extra_comments, quality score, and native price.

    Args:
        processed_grid (list): The list of processed sale data tuples for the current batch.
                               Each tuple: (date, media_q, sleeve_q, price_float,
                                           native_price_str, score, comment)
        artist (str or None): The extracted artist name for this batch.
        album (str or None): The extracted album title for this batch.
        label (str or None): The extracted label for this batch.
        extra_comments (str or None): The extracted extra comments/format details.
    """
    # Read existing ML data (now just a list of sales)
    existing_ml_sales = read_ml_data()

    # Create a set of identifiers for existing sales for quick lookup
    existing_identifiers = set()
    for sale in existing_ml_sales:
        identifier = (
            sale.get('date', ''),
            sale.get('artist', ''),
            sale.get('album', ''),
            sale.get('label', ''),
            sale.get('extra_comments', ''),
            round(sale.get('quality', 0.0), 5),
            sale.get('native_price', '')
        )
        existing_identifiers.add(identifier)

    new_ml_sales_to_add = []
    current_batch_identifiers = set()

    for row in processed_grid:
        if len(row) >= 7:
            sale_date = row[0]
            native_price_from_row = row[4]
            quality_score = row[5]
            inflation_adjusted_price = row[3]

            current_sale_identifier = (
                sale_date or '',
                artist or '',
                album or '',
                label or '',
                extra_comments or '',
                round(quality_score, 5),
                native_price_from_row or ''
            )

            if current_sale_identifier in existing_identifiers:
                continue
            if current_sale_identifier in current_batch_identifiers:
                continue

            current_batch_identifiers.add(current_sale_identifier)

            sale_data_dict = {
                'date': sale_date,
                'quality': quality_score,
                'price': inflation_adjusted_price,
                'native_price': native_price_from_row,
                'artist': artist,
                'album': album,
                'label': label,
                'extra_comments': extra_comments
            }
            new_ml_sales_to_add.append(sale_data_dict)
        else:
            print(f"Warning: Skipping row with unexpected structure for ML data: {row}", file=sys.stderr)

    if new_ml_sales_to_add:
        # Append new unique data to existing data
        combined_ml_sales = existing_ml_sales + new_ml_sales_to_add

        # Save the combined data
        write_ml_data(combined_ml_sales)

        print(f"Info: Saved {len(new_ml_sales_to_add)} NEW unique data point(s) for ML training.",
              file=sys.stderr)
        print(f"Info: Data saved for Record: Artist='{artist}', Album='{album}', Label='{label}', Extra='{extra_comments}'.",
              file=sys.stderr)
    else:
        print(f"Info: No new unique sales found in pasted data for Record: Artist='{artist}', Album='{album}', Label='{label}', Extra='{extra_comments}'. ML data file was not updated.", file=sys.stderr)

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

def extract_tuples(processed_grid):
    # gets dates and comments from the processed_grid to go in the output
    dates = []
    comments = []
    qualities = []
    prices = []

    # Ensure the loop handles the structure correctly, especially after filtering
    for row in processed_grid:
        dates.append(row[0]) # Date is index 0
        qualities.append(row[5])  # Quality column (score)
        prices.append(row[3])  # Price column

        if row[6]:
            comments.append(row[6]) # if no comment, add blank
        else:
            comments.append("") # Default comment

    return qualities, prices, dates, comments

def merge_and_deduplicate_grids(grid1, grid2):
    """
    Helper function to merge two grids of data rows and remove duplicates.
    Assumes rows are lists that can be converted to tuples for set operations.
    """
    seen_rows = set()
    merged_grid = []

    # Process first grid
    for row_list in grid1:
        # Ensure row_list elements are hashable for tuple conversion, handle None for comments
        # Example: (date, media, sleeve, price_float, native_price, score, comment or "")
        # Assuming price_float can be None, but tuple() handles it.
        # Comments (row_list[6]) can be None, ensure it's handled if it causes issues with tuple.
        # A common practice is to ensure all elements are consistently typed or handled before tuple conversion.
        # For simplicity here, direct tuple conversion is used.
        try:
            row_tuple = tuple(row_list)
        except TypeError:
            # Fallback if a row element is not hashable (e.g. a list itself)
            # This shouldn't happen if processed_grid rows are flat lists/tuples of primitives.
            # For robustness, convert to string representation or handle specific unhashable types.
            # Here, we'll assume rows are simple enough for direct tuple conversion.
            # If issues arise, this is a point for deeper inspection of row contents.
            row_tuple = tuple(str(item) for item in row_list) # A basic fallback

        if row_tuple not in seen_rows:
            merged_grid.append(list(row_list)) # Store as list of lists
            seen_rows.add(row_tuple)

    # Process second grid
    for row_list in grid2:
        try:
            row_tuple = tuple(row_list)
        except TypeError:
            row_tuple = tuple(str(item) for item in row_list)

        if row_tuple not in seen_rows:
            merged_grid.append(list(row_list))
            seen_rows.add(row_tuple)
    return merged_grid

def manage_processed_grid(discogs_data, start_date, points_to_delete_json, add_data_str):
    """
    Manages the lifecycle of the processed sales data grid.

    This includes parsing new data from `discogs_data`, optionally merging
    it with previously saved data if `add_data_str` is 'True', deleting
    specified points, and then saving the resulting grid. If `discogs_data`
    is empty, it loads and works with the saved grid.

    Args:
        discogs_data (str): Raw text data from Discogs sales history.
        start_date (str): The start date ('YYYY-MM-DD') for filtering sales data,
                          used when parsing `discogs_data`.
        points_to_delete_json (str): JSON string array of points selected for deletion.
        add_data_str (str): String ('True' or 'False'). If 'True' and `discogs_data`
                            is provided, newly parsed data is merged with saved data.
                            If `discogs_data` is not provided, this flag is ignored
                            and only saved data is loaded.

    Returns:
        tuple: A tuple containing:
            - processed_grid (list): The final list of processed sale data tuples.
            - deleted_count (int): The number of points removed from the grid.
            - status_message_from_parsing (str or None): Status message from
                                                         `make_processed_grid` if new
                                                         data was parsed, otherwise None.

    Raises:
        ValueError: If no data points are available after all operations, or if
                    initial parsing of `discogs_data` yields a critical error message
                    (e.g., "No Discogs Data in text box") and `add_data_str` is not 'True'.
    """
    current_grid_for_processing = []
    status_message_from_parsing = None

    if discogs_data:
        # New data is provided
        newly_parsed_data, status_message_from_parsing = make_processed_grid(discogs_data, start_date)

        # Handle critical parsing messages early if not adding to existing data
        if status_message_from_parsing and "No Discogs Data in text box" in status_message_from_parsing:
            if add_data_str != "True": # If not adding, and parsing failed critically, raise error
                raise ValueError(status_message_from_parsing)
            # If adding, newly_parsed_data might be empty, but we'll proceed to merge with saved.

        if add_data_str == "True":
            saved_grid = read_save_value("processed_grid", [])
            # Merge newly parsed data (which might be empty if parsing had issues but add_data is True)
            # with the saved grid.
            current_grid_for_processing = merge_and_deduplicate_grids(newly_parsed_data, saved_grid)
        else:
            # Use only the newly parsed data
            current_grid_for_processing = newly_parsed_data
    else:
        # No new discogs_data, load entirely from save.
        # add_data_str is effectively ignored here as there's no new data to "add" to saved data.
        current_grid_for_processing = read_save_value("processed_grid", [])
        # status_message_from_parsing remains None

    # Now, delete points from the assembled grid
    final_grid, deleted_count = delete_points(points_to_delete_json, current_grid_for_processing)

    # Check if the grid is empty after all operations
    if not final_grid:
        # If a critical parsing error occurred and we didn't load/merge any other data
        if status_message_from_parsing and "No Discogs Data in text box" in status_message_from_parsing:
            raise ValueError(status_message_from_parsing)
        # Otherwise, it's a general "no data" error after all steps.
        raise ValueError("No data points available for analysis after processing, loading, or deletion.")

    # Save the final grid
    write_save_value(final_grid, "processed_grid")

    return final_grid, deleted_count, status_message_from_parsing

def extract_record_metadata(clipboard_content):
    """
    Extracts Artist, Album, Label, and Extra Comments from the line
    following "Recent Sales History".

    Args:
        clipboard_content (str): The raw text pasted from Discogs.

    Returns:
        tuple: (artist, album, label, extra_comments) strings,
               where any element can be None if it cannot be parsed.
    """
    lines = clipboard_content.splitlines()
    recent_sales_history_index = -1

    for i, line in enumerate(lines):
        if "Recent Sales History" in line:
            recent_sales_history_index = i
            break

    if recent_sales_history_index == -1 or recent_sales_history_index + 1 >= len(lines):
        print("Warning: 'Recent Sales History' marker not found or no line follows it. Cannot extract metadata.", file=sys.stderr)
        return None, None, None, None

    metadata_line = lines[recent_sales_history_index + 1].strip()

    if not metadata_line:
        print("Warning: Metadata line after 'Recent Sales History' is empty.", file=sys.stderr)
        return None, None, None, None

    artist = None
    album = None
    label = None
    extra_comments = None

    last_dash_index = metadata_line.rfind(" - ")

    if last_dash_index != -1:
        artist = metadata_line[:last_dash_index].strip()
        # Part after "Artist - "
        title_label_extra_part = metadata_line[last_dash_index + len(" - "):].strip()

        last_open_paren_index = title_label_extra_part.rfind("(")
        last_close_paren_index = title_label_extra_part.rfind(")")

        if last_open_paren_index != -1 and last_close_paren_index != -1 and last_close_paren_index > last_open_paren_index:
            # Album is between " - " and last "("
            album = title_label_extra_part[:last_open_paren_index].strip()
            # Label is between last "(" and last ")"
            extracted_label_raw = title_label_extra_part[last_open_paren_index + 1:last_close_paren_index].strip()

            # --- New Label Cleaning Logic ---
            if extracted_label_raw:
                # Split by comma, strip whitespace from each part
                label_parts = [part.strip() for part in extracted_label_raw.split(',')]
                # Get unique parts
                unique_label_parts = list(dict.fromkeys(label_parts)) # Preserves order
                # If all unique parts are the same (meaning the original was like "Label, Label, Label")
                if len(unique_label_parts) == 1:
                    label = unique_label_parts[0] # Use the single unique part
                else:
                    # If there are multiple different parts, join them back with ", "
                    label = ", ".join(unique_label_parts)

            # Extra comments are after last ")"
            if last_close_paren_index + 1 < len(title_label_extra_part):
                extra_comments = title_label_extra_part[last_close_paren_index + 1:].strip()
                if extra_comments.endswith('*'):
                    extra_comments = extra_comments[:-1].strip()
        else:
            # No parentheses for label, so the whole part after "Artist - " is the album
            album = title_label_extra_part.strip()
            # Label and extra_comments remain None
    else:
        # No " - " found, assume the whole line is the album/title
        # According to user definition, artist cannot be found.
        # We can try to parse label and extra_comments if parentheses exist
        last_open_paren_index = metadata_line.rfind("(")
        last_close_paren_index = metadata_line.rfind(")")

        if last_open_paren_index != -1 and last_close_paren_index != -1 and last_close_paren_index > last_open_paren_index:
            album = metadata_line[:last_open_paren_index].strip() # Part before label is album
            label = metadata_line[last_open_paren_index + 1:last_close_paren_index].strip()
            if last_close_paren_index + 1 < len(metadata_line):
                extra_comments = metadata_line[last_close_paren_index + 1:].strip()
                if extra_comments.endswith('*'):
                    extra_comments = extra_comments[:-1].strip()
        else:
            # No " - " and no parentheses for label, so the whole line is the album
            album = metadata_line.strip()
            # Artist, Label, extra_comments remain None


    # The previous specific regex cleanups for 'album' that might have removed
    # format details are no longer strictly necessary here, as 'extra_comments'
    # is now intended to capture those.
    # You might still want a generic cleanup for album if needed.
    if album:
        album = album.strip() # Ensure album is stripped

    return artist, album, label, extra_comments