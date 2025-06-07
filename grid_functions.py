import re
from datetime import datetime
import math
import json
import sys
from persistence import read_save_value, write_save_value, read_ml_data, write_ml_data
import discogs_client
from discogs_secrets import DISCOGS_USER_TOKEN # Import your token

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
            print("Could not find 'Order Date' header row.")
            return []

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


def make_processed_grid(clipboard_content, start_date_str_param, discogs_release_id=None,
                        api_data=None):  # Add api_data
    """
    Parses raw text data (presumably pasted from Discogs sales history)
    into a structured grid format.
    Now also accepts api_data to be passed to machine_learning_save.
    Metadata (artist, album, etc.) is expected to come from api_data.
    """
    status_message = None
    processed_grid = []
    ml_save_setting = True

    if discogs_release_id:
        print(f"GRID_FUNCTIONS.PY (make_processed_grid): Received Discogs Release ID: {discogs_release_id}")
    if api_data:
        print(
            f"GRID_FUNCTIONS.PY (make_processed_grid): API data provided for {api_data.get('api_artist')} - {api_data.get('api_title')}")
    else:
        if discogs_release_id:  # Only print "no API data" if an ID was actually expected
            print(f"GRID_FUNCTIONS.PY (make_processed_grid): No API data provided for Release ID: {discogs_release_id}")

    # --- Parse Sales Data from clipboard_content ---
    if clipboard_content and clipboard_content.strip():  # Only parse if there's actual content
        if "Order Date" not in clipboard_content or "Change Currency" not in clipboard_content:
            # This is a problem for parsing sales data.
            # If API data is present, we might still want to save it.
            # For now, if sales data is malformed, we'll return an error for sales parsing,
            # but ML saving might still proceed if api_data is available (handled by manage_processed_grid).
            status_message = "Sales data is malformed or missing headers."
            print(f"Warning: {status_message}", file=sys.stderr)
            # We don't return immediately if api_data might be present,
            # as manage_processed_grid might still want to save API context.
            # processed_grid will remain empty.
        else:
            try:
                start_date_obj = datetime.strptime(start_date_str_param, '%Y-%m-%d').date()
            except ValueError:
                return [], "Invalid start_date format. Please use YYYY-MM-DD."  # Critical error for parsing

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
                if entry_error:
                    print(entry_error, file=sys.stderr)  # Log individual entry errors

                if lines_consumed == 0:  # Safety break for infinite loop
                    print(f"Warning: _parse_single_entry consumed 0 lines at index {i}. Incrementing by 1.",
                          file=sys.stderr)
                    i += 1
                else:
                    i += lines_consumed
    else:
        print("GRID_FUNCTIONS.PY (make_processed_grid): No sales data in clipboard_content to parse.")
        # processed_grid remains empty

    # --- Save to ML data ---
    # We save if ml_save_setting is True AND we have either:
    # 1. API data (even if no sales were parsed from clipboard)
    # 2. Parsed sales data (even if no API data, e.g., API call failed or no ID given)
    if ml_save_setting:
        if api_data or processed_grid:  # Only attempt to save if there's *something* to save
            try:
                # The old extract_record_metadata(clipboard_content) call is REMOVED.
                # Artist, album, label, etc., for ML save will now come from api_data
                # or be None if api_data is not available.
                machine_learning_save(processed_grid, discogs_release_id, api_data)
            except Exception as e:
                print(f"Error during machine_learning_save call: {e}", file=sys.stderr)
        else:
            print(
                "GRID_FUNCTIONS.PY (make_processed_grid): No API data and no processed sales data. Nothing to save for ML.",
                file=sys.stderr)

    # If status_message was set due to malformed sales data but API data was processed,
    # we might want to clear it or make it more nuanced.
    # For now, it reflects the sales parsing part.
    if not processed_grid and status_message == "Sales data is malformed or missing headers.":
        # If we ended up with no sales grid due to bad sales data, but API data might have been saved,
        # the status_message should reflect that sales parsing failed.
        pass  # Keep the status_message as is.

    return processed_grid, status_message

def machine_learning_save(processed_grid, discogs_release_id=None, api_data=None):
    """
    Saves or updates release information and its associated sales data to the ML data file.
    The ML data is structured as a dictionary keyed by discogs_release_id.
    Each release entry contains API metadata and a list of its sales history.
    """
    all_releases_data = read_ml_data()  # Expects a dictionary, or {} if new/empty

    if not isinstance(all_releases_data, dict):
        print("Warning: ML data file is not a dictionary. Initializing as empty. "
              "If you have old list-based data, it needs migration.", file=sys.stderr)
        all_releases_data = {}

    if not discogs_release_id:
        if processed_grid:
            print("Warning: Sales data provided in processed_grid, but no discogs_release_id. "
                  "These sales will not be saved in the new release-centric ML data structure.", file=sys.stderr)
        else:
            print("Info (machine_learning_save): No discogs_release_id and no sales data. Nothing to save.",
                  file=sys.stderr)
        return  # Cannot proceed without a release ID for the new structure

    release_key = str(discogs_release_id)
    new_sales_added_count = 0

    # Get or create the entry for this release
    release_entry = all_releases_data.get(release_key)

    if release_entry is None:
        release_entry = {}
        if api_data:
            release_entry.update(api_data)
        release_entry['sales_history'] = []
        all_releases_data[release_key] = release_entry
        print(f"Info: Creating new entry for Release ID: {release_key}", file=sys.stderr)
    else:
        # Update existing API data if new API data is provided
        if api_data:
            # You might want a more sophisticated merge strategy here if needed,
            # e.g., only update if new data is different or more complete.
            # For now, a simple update will overwrite existing api_ fields with new ones.
            release_entry.update(api_data)
            print(f"Info: Updating API data for existing Release ID: {release_key}", file=sys.stderr)
        if 'sales_history' not in release_entry or not isinstance(release_entry['sales_history'], list):
            release_entry['sales_history'] = []  # Ensure sales_history list exists and is a list

    # De-duplicate sales within this specific release's sales_history
    # Create a set of identifiers for sales already in this release's history
    existing_sale_identifiers_for_release = set()
    for sale in release_entry['sales_history']:
        # Key for a sale: (date, quality_score_rounded, native_price)
        sale_identifier = (
            sale.get('date', ''),
            round(sale.get('quality', 0.0), 5),
            sale.get('native_price', '')
        )
        existing_sale_identifiers_for_release.add(sale_identifier)

    # For de-duping sales within the current processed_grid batch for this release
    current_batch_sale_identifiers = set()

    if processed_grid:
        for row in processed_grid:
            if len(row) >= 7:
                sale_date = row[0]
                native_price_from_row = row[4]
                quality_score = row[5]
                inflation_adjusted_price = row[3]
                sale_specific_comment = row[6]

                current_sale_de_dup_key = (
                    sale_date or '',
                    round(quality_score, 5),
                    native_price_from_row or ''
                )

                if current_sale_de_dup_key in existing_sale_identifiers_for_release or \
                        current_sale_de_dup_key in current_batch_sale_identifiers:
                    continue  # Skip if already exists for this release or in current batch

                current_batch_sale_identifiers.add(current_sale_de_dup_key)

                sale_dict = {
                    'date': sale_date,
                    'quality': quality_score,
                    'price': inflation_adjusted_price,
                    'native_price': native_price_from_row,
                    'sale_comment': sale_specific_comment
                }
                release_entry['sales_history'].append(sale_dict)
                new_sales_added_count += 1
            else:
                print(f"Warning: Skipping row with unexpected structure for ML data: {row}", file=sys.stderr)
    elif api_data:  # No sales in processed_grid, but API data was provided (or updated)
        print(f"Info: API data for Release ID {release_key} processed. No new sales from current input.",
              file=sys.stderr)

    if new_sales_added_count > 0:
        write_ml_data(all_releases_data)
        artist_name_for_log = release_entry.get('api_artist', 'N/A')
        album_title_for_log = release_entry.get('api_title', 'N/A')
        print(f"Info: Added {new_sales_added_count} new sales to Release ID '{release_key}': "
              f"Artist='{artist_name_for_log}', Album='{album_title_for_log}'. ML data updated.", file=sys.stderr)
    elif api_data and release_key in all_releases_data and not processed_grid:
        # This case means API data was potentially updated for an existing release, but no new sales were added.
        # We should still save if the release_entry itself was modified (e.g. API data updated)
        write_ml_data(all_releases_data)  # Save if API data was added/updated
        artist_name_for_log = release_entry.get('api_artist', 'N/A')
        album_title_for_log = release_entry.get('api_title', 'N/A')
        print(
            f"Info: API data for Release ID '{release_key}' (Artist='{artist_name_for_log}', Album='{album_title_for_log}') "
            f"processed/updated. No new sales added in this batch. ML data updated.", file=sys.stderr)
    else:
        # This covers cases like:
        # - No new sales and no API data to update for an existing release.
        # - Release ID was new, API data was fetched, but processed_grid was empty (already logged).
        artist_name_for_log = release_entry.get('api_artist', 'N/A') if release_entry else 'N/A'
        album_title_for_log = release_entry.get('api_title', 'N/A') if release_entry else 'N/A'
        print(f"Info: No new unique sales found in pasted data for Release ID '{release_key}': "
              f"Artist='{artist_name_for_log}', Album='{album_title_for_log}'. "
              f"ML data file not updated with new sales for this release.", file=sys.stderr)

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

def manage_processed_grid(discogs_data, start_date, points_to_delete_json, add_data_str, discogs_release_id=None):
    """
    Manages the lifecycle of the processed sales data grid.
    ... (rest of docstring)
    """
    current_grid_for_processing = []
    status_message_from_parsing = None
    api_data_for_release = None # Initialize api_data_for_release

    if discogs_release_id:
        print(f"GRID_FUNCTIONS.PY (manage_processed_grid): Received Discogs Release ID: {discogs_release_id}")
        print(f"Attempting to fetch API data for release ID: {discogs_release_id}")
        api_data_for_release = fetch_api_data(discogs_release_id) # Call the new function
        if api_data_for_release:
            print(f"Successfully fetched API data for: {api_data_for_release.get('api_artist')} - {api_data_for_release.get('api_title')}")
        else:
            print(f"Failed to fetch API data or ID not found for {discogs_release_id}.")
    else:
        print("GRID_FUNCTIONS.PY (manage_processed_grid): No Discogs Release ID provided.")


    if discogs_data:
        newly_parsed_data, status_message_from_parsing = make_processed_grid(
            discogs_data,
            start_date,
            discogs_release_id,
            api_data_for_release # --- NEW: Pass api_data_for_release ---
        )
        if status_message_from_parsing and "No Discogs Data in text box" in status_message_from_parsing:
            if add_data_str != "True":
                raise ValueError(status_message_from_parsing)

        if add_data_str == "True":
            saved_grid = read_save_value("processed_grid", [])
            current_grid_for_processing = merge_and_deduplicate_grids(newly_parsed_data, saved_grid)
        else:
            current_grid_for_processing = newly_parsed_data
    else: # No new discogs_data pasted
        current_grid_for_processing = read_save_value("processed_grid", [])
        # If no sales data pasted, but we have API data, we still need to trigger ML save
        if api_data_for_release and discogs_release_id:
             print("Info: No sales data pasted, but API data is available. Proceeding to save API data context for ML.")
             # Call make_processed_grid with empty sales data but with API data
             # This will ensure machine_learning_save is called with the API data.
             _, status_message_from_parsing = make_processed_grid(
                "", # Empty sales data
                start_date,
                discogs_release_id,
                api_data_for_release
            )
        # else: no sales data pasted and no API data, so current_grid_for_processing (from save file) is used.


    final_grid, deleted_count = delete_points(points_to_delete_json, current_grid_for_processing)

    if not final_grid:
        # If API data was fetched but sales parsing (or loading from file) resulted in an empty grid,
        # we might still want to proceed if the goal is to save the API context.
        # However, the current `vin.py` expects `processed_grid` to have data for charting.
        # This needs careful consideration based on whether an ID-only entry (no sales) is useful.
        # For now, maintaining original behavior: if final_grid is empty, it's an issue for charting.
        if not (api_data_for_release and not discogs_data): # Allow empty final_grid if it's an API-only call with no sales data
            if status_message_from_parsing and "No Discogs Data in text box" in status_message_from_parsing:
                raise ValueError(status_message_from_parsing)
            raise ValueError("No data points available for analysis after processing, loading, or deletion.")


    write_save_value(final_grid, "processed_grid") # This saves the sales data for charting

    return final_grid, deleted_count, status_message_from_parsing

def fetch_api_data(release_id):
    """
    Fetches detailed information for a given release ID from the Discogs API,
    including the original year from its master release.

    Args:
        release_id (str or int): The Discogs release ID.

    Returns:
        dict: A dictionary containing extracted release information (year, artist,
              title, genres, styles, country, label, catno, format_descriptions, notes,
              community_have, community_want, community_rating_average,
              community_rating_count, master_id, api_original_year),
              or None if an error occurs or ID is not found.
    """
    if not release_id:
        return None

    try:
        # Initialize the Discogs client
        d = discogs_client.Client('VinylPriceCalculator/0.1', user_token=DISCOGS_USER_TOKEN)
        release = d.release(int(release_id))  # Ensure release_id is an integer

        api_data = {}

        # --- Extract Key Information from the specific release ---
        api_data['api_year'] = getattr(release, 'year', None)
        api_data['api_title'] = getattr(release, 'title', None)

        if release.artists:
            api_data['api_artist'] = getattr(release.artists[0], 'name', None)
        else:
            api_data['api_artist'] = None

        api_data['api_genres'] = getattr(release, 'genres', [])
        api_data['api_styles'] = getattr(release, 'styles', [])
        api_data['api_country'] = getattr(release, 'country', None)

        if release.labels:
            api_data['api_label'] = getattr(release.labels[0], 'name', None)
            api_data['api_catno'] = getattr(release.labels[0], 'catno', None)
        else:
            api_data['api_label'] = None
            api_data['api_catno'] = None

        format_descriptions = []
        if release.formats:
            for fmt in release.formats:
                if fmt.get('descriptions'):
                    format_descriptions.extend(fmt['descriptions'])
        api_data['api_format_descriptions'] = list(set(format_descriptions))

        api_data['api_notes'] = release.data.get('notes', None)

        community_data = release.data.get('community', {})
        api_data['api_community_have'] = community_data.get('have', None)
        api_data['api_community_want'] = community_data.get('want', None)

        rating_data = community_data.get('rating', {})
        api_data['api_community_rating_average'] = rating_data.get('average', None)
        api_data['api_community_rating_count'] = rating_data.get('count', None)

        # --- Fetch Master Release ID and Original Year ---
        master_id = getattr(release, 'master_id', None)
        print(f"Attempt 1: master_id from release object attribute for {release_id}: {master_id}")

        if not master_id or master_id == 0:
            master_id_from_data = release.data.get('master_id')
            print(f"Attempt 2: master_id from release.data for {release_id}: {master_id_from_data}")
            if master_id_from_data and master_id_from_data != 0:
                master_id = master_id_from_data
            else:
                master_id = None

        api_data['api_master_id'] = master_id
        api_data['api_original_year'] = None  # Initialize

        if master_id:
            print(f"Found Master ID: {master_id} for Release ID: {release_id}. Fetching master release...")
            try:
                master_release = d.master(int(master_id))  # Ensure master_id is int for the call

                # --- NEW: Explicitly refresh the master_release object to get all data ---
                print(f"  Attempting to refresh master_release object (ID: {master_id}) to fetch full data...")
                master_release.refresh()
                print(f"  Refresh complete for master_release object (ID: {master_id}).")
                # --- END OF REFRESH ---

                # --- DETAILED INSPECTION OF master_release OBJECT (POST-REFRESH) ---
                print(f"  INSPECTION (POST-REFRESH): Type of master_release object: {type(master_release)}")
                print(f"  INSPECTION (POST-REFRESH): Type of master_release.data: {type(master_release.data)}")
                if isinstance(master_release.data, dict):
                    print(
                        f"  INSPECTION (POST-REFRESH): Keys in master_release.data: {list(master_release.data.keys())}")
                    if 'year' in master_release.data:
                        print(
                            f"  INSPECTION (POST-REFRESH): Value of master_release.data['year']: {master_release.data.get('year')}")
                    else:
                        print(f"  INSPECTION (POST-REFRESH): 'year' key NOT FOUND in master_release.data.")
                else:
                    print(f"  INSPECTION (POST-REFRESH): master_release.data is not a dictionary.")
                # --- END OF DETAILED INSPECTION (POST-REFRESH) ---

                # Attempt 1: Get year from master_release attribute
                original_year_from_attr = getattr(master_release, 'year', None)
                print(
                    f"  Master Year (from attribute, post-refresh): {original_year_from_attr} for Master ID: {master_id}")

                # Attempt 2: If attribute access fails or gives 0/None, try from master_release.data
                if not original_year_from_attr or original_year_from_attr == 0:
                    original_year_from_data = master_release.data.get('year')
                    print(
                        f"  Master Year (from .data.get('year'), post-refresh): {original_year_from_data} for Master ID: {master_id}")
                    if original_year_from_data and original_year_from_data != 0:
                        api_data['api_original_year'] = original_year_from_data
                    else:
                        if original_year_from_attr and original_year_from_attr != 0:
                            api_data['api_original_year'] = original_year_from_attr
                        else:
                            api_data['api_original_year'] = None
                else:
                    api_data['api_original_year'] = original_year_from_attr

                # Logging based on the final api_data['api_original_year']
                if api_data['api_original_year'] is not None and api_data['api_original_year'] != 0:
                    print(
                        f"Successfully fetched original year: {api_data['api_original_year']} for Master ID: {master_id}")
                elif api_data['api_original_year'] == 0:
                    print(
                        f"Master ID {master_id} (Release ID: {release_id}) has 'year' attribute as 0 (or resolved to 0).",
                        file=sys.stderr)
                else:
                    print(
                        f"Master ID {master_id} (Release ID: {release_id}) found, but 'year' attribute is missing or None on the master_release object (after checking attribute and .data, post-refresh).",
                        file=sys.stderr)

            except discogs_client.exceptions.HTTPError as master_http_err:
                if master_http_err.status_code == 404:
                    print(f"Discogs API Error: Master ID {master_id} (for Release ID: {release_id}) not found (404).",
                          file=sys.stderr)
                else:
                    print(
                        f"Discogs API HTTP Error for Master ID {master_id} (for Release ID: {release_id}): {master_http_err}",
                        file=sys.stderr)
            except Exception as master_e:
                print(
                    f"An unexpected error occurred while fetching Master ID {master_id} (for Release ID: {release_id}): {master_e}",
                    file=sys.stderr)
        else:
            print(f"No valid Master ID found on Release ID: {release_id} after checking object attribute and .data.")
        # --- End of Master Release Fetch ---

        print(f"Successfully fetched API data for release ID: {release_id}")
        return api_data

    except discogs_client.exceptions.HTTPError as http_err:
        if http_err.status_code == 404:
            print(f"Discogs API Error: Release ID {release_id} not found (404).", file=sys.stderr)
        elif http_err.status_code == 401:
            print(f"Discogs API Error: Unauthorized (401). Check your User Token.", file=sys.stderr)
        else:
            print(f"Discogs API HTTP Error for ID {release_id}: {http_err}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred while fetching API data for ID {release_id}: {e}", file=sys.stderr)
        return None
