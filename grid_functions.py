# /home/stigoftdump/PycharmProjects/PythonProject/vinylprice/grid_functions.py
import re
from datetime import datetime
import math
import json
import sys
from persistence import read_save_value, write_save_value, read_ml_data, write_ml_data
import discogs_client
from discogs_secrets import DISCOGS_USER_TOKEN  # Import your token

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
real_prices = [
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
    2024: 0.06,  # Placeholder, update with actual when available
}


def adjust_for_inflation(price, sale_year):
    """
    Adjusts a historical price to its equivalent value in a target year (currently 2024) using inflation rates.

    Args:
        price (float): The original price from the sale year.
        sale_year (int): The year the sale occurred.

    Returns:
        float: The inflation-adjusted price, rounded to 2 decimal places.
               Returns the original price if adjustment cannot be fully completed
               due to missing inflation rates.
    """
    target_inflation_year = 2024  # Adjust all prices to 2024 values
    adjustment_factor = 1.0

    # Iterate through years from sale_year up to the target year
    for year in range(sale_year, target_inflation_year + 1):  # Iterate up to and including target_inflation_year
        if year in inflation_rates:
            adjustment_factor *= (1 + inflation_rates[year])
        else:
            # If inflation rate for a year is missing, we cannot adjust accurately for that year onwards
            if year <= target_inflation_year:  # Only warn if it's a year we need for adjustment
                print(f"Warning: Missing inflation rate for year {year}. Inflation adjustment may be incomplete.",
                      file=sys.stderr)
            break  # Stop adjustment if rate is missing for a year in the range

    new_price = round(price * adjustment_factor, 2)

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

    if record_value == 0:  # If record quality is unknown/unmapped, score is 0
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
    if pred_price < 0:  # Handle negative predicted prices
        return 0.0
    if pred_price < 42.48:
        # Finds the closest price using the min function with a custom key
        foundprice = min(real_prices, key=lambda x: abs(x - pred_price))
    elif pred_price < 100:
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
            print("Warning: Could not find 'Order Date' header row in sales data.", file=sys.stderr)
            return []  # Return empty list, make_processed_grid will handle this

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
            print(f"Warning: Could not convert user price '{price_user_str}' to float.", file=sys.stderr)
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
    if part_length >= 5:  # linux
        # are wet the end?
        if iteration + 1 < len(relevant_rows):
            if relevant_rows[iteration + 1].strip().startswith(
                    'Comments:'):  # does the next line start with "comments:"?
                lines_to_skip_ahead = 1
    elif part_length == 4:  # windows
        # the end?
        if iteration + 1 < len(relevant_rows):  # Native price line
            lines_to_skip_ahead = 1
            if iteration + 2 < len(relevant_rows):  # Comments line
                if relevant_rows[iteration + 2].strip().startswith(
                        'Comments:'):  # does the next line start with "comments:"?
                    lines_to_skip_ahead = 2

    return lines_to_skip_ahead


def extract_native_price(part_length, linux_date_or_price_part, relevant_rows, iteration):
    """
    Extracts the native (original currency) price string.

    Args:
        part_length (int): Number of tab-separated parts in the current primary row.
        linux_date_or_price_part (str): The 5th part from the primary row (Linux),
                                       which contains the native price.
        relevant_rows (list): List of all relevant sales data rows.
        iteration (int): The current index in relevant_rows for the primary row.

    Returns:
        str or None: The native price string, or None if not found.
    """
    native_price = None
    if part_length >= 5:  # linux
        native_price = linux_date_or_price_part.strip()  # This is data_parts[4]
    elif part_length == 4:  # windows
        if iteration + 1 < len(relevant_rows):
            native_price = relevant_rows[iteration + 1].strip()  # Price is on the next line
    # else: native_price remains None

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
        # else: # No need for a warning here, it's normal if no comments line follows
        # print(f"Warning: Windows format, line i+1 missing") # This warning was a bit noisy

    return comment_str_out


def _parse_single_entry(relevant_rows, current_index, date_pattern, start_date_obj):
    """
    Attempts to parse a single Discogs sales entry starting from current_index.
    A single entry might span multiple lines due to format differences (Linux vs. Windows paste).

    Args:
        relevant_rows (list): List of strings, each a line from the relevant data block.
        current_index (int): The index in relevant_rows to start parsing from.
        date_pattern (re.Pattern): Compiled regex pattern to match dates.
        start_date_obj (datetime.date): The parsed start date for filtering.

    Returns:
        tuple: (parsed_data_tuple, lines_consumed, error_message_for_entry)
               - parsed_data_tuple (tuple or None): The structured tuple of the parsed entry
                 (date_str, quality1_str, quality2_str, price_float, native_price_str, score, comment_str),
                 or None if parsing failed or no entry found.
               - lines_consumed (int): Number of lines from relevant_rows processed for this attempt.
                                       Will be at least 1.
               - error_message_for_entry (str or None): Error message if parsing this specific
                                                        entry failed, otherwise None.
    """
    if current_index >= len(relevant_rows):
        return None, 0, None

    row = relevant_rows[current_index]
    match = date_pattern.match(row)

    if not match:
        return None, 1, None  # No date found at the start of this line

    try:
        data_parts = row.split('\t')
        part_length = len(data_parts)

        date_str_raw = data_parts[0].strip()
        date_obj = datetime.strptime(date_str_raw, '%Y-%m-%d').date()

        if date_obj < start_date_obj:
            return None, 1, None  # Entry is too old

        date_str_out = date_str_raw

        if part_length < 3:
            raise ValueError(f"Row '{row[:50]}...' started with a date but had less than 3 tab-separated parts.")
        quality1_str_raw = data_parts[1]
        quality2_str_raw = data_parts[2]

        price_string_from_data = data_parts[3] if part_length > 3 else ""
        price_float_out = extract_price(price_string_from_data, date_obj)

        lines_to_skip_ahead = calculate_line_skip(part_length, current_index, relevant_rows)

        # For native price, pass the part that might contain it (for Linux)
        native_price_data_part_for_linux = data_parts[4].strip() if part_length > 4 else ""
        native_price_str_out = extract_native_price(part_length, native_price_data_part_for_linux, relevant_rows,
                                                    current_index)

        comment_str_out = extract_comments(part_length, relevant_rows, current_index)
        score_out = calculate_score(quality1_str_raw, quality2_str_raw)

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
        error_msg = f"Error processing entry at line {current_index} ('{row[:70]}...'): {e}"
        return None, 1, error_msg


def make_processed_grid(clipboard_content, start_date_str_param, discogs_release_id=None,
                        api_data=None):
    """
    Parses raw text data (presumably pasted from Discogs sales history)
    into a structured grid format. Also triggers saving of API data and parsed sales
    to the ML data store if applicable.

    Args:
        clipboard_content (str): Raw text from Discogs sales history.
        start_date_str_param (str): Start date for filtering sales ('YYYY-MM-DD').
        discogs_release_id (str, optional): The Discogs release ID.
        api_data (dict, optional): Pre-fetched API data for the release.

    Returns:
        tuple: (processed_grid, status_message)
               - processed_grid (list): List of parsed sales data tuples.
               - status_message (str or None): Message indicating parsing status or errors.
    """
    status_message = None
    processed_grid = []
    # ml_save_setting = True # This variable is not used, can be removed

    if discogs_release_id:
        print(f"Info: Processing sales data for Release ID: {discogs_release_id}")
    if api_data:
        print(
            f"Info: API data available for {api_data.get('api_artist', 'N/A')} - {api_data.get('api_title', 'N/A')}")

    if clipboard_content and clipboard_content.strip():
        # Basic check for essential markers. More robust parsing is in get_relevant_rows.
        if "Order Date" not in clipboard_content:  # Simplified check, get_relevant_rows handles more detail
            status_message = "Sales data might be malformed or missing 'Order Date' header."
            print(f"Warning: {status_message}", file=sys.stderr)
            # Continue to allow API data saving even if sales parsing fails
        else:
            try:
                start_date_obj = datetime.strptime(start_date_str_param, '%Y-%m-%d').date()
            except ValueError:
                return [], "Invalid start_date format. Please use YYYY-MM-DD."

            relevant_rows = get_relevant_rows(clipboard_content)
            if not relevant_rows and "Warning: Could not find 'Order Date' header row" in status_message if status_message else False:
                # If get_relevant_rows returned empty due to missing header, status is already set.
                pass
            elif not relevant_rows:
                status_message = "No relevant sales rows found after header processing."
                print(f"Info: {status_message}", file=sys.stderr)

            date_pattern = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})")
            i = 0
            while i < len(relevant_rows):
                parsed_tuple, lines_consumed, entry_error = _parse_single_entry(
                    relevant_rows, i, date_pattern, start_date_obj
                )
                if parsed_tuple:
                    processed_grid.append(parsed_tuple)
                if entry_error:
                    print(entry_error, file=sys.stderr)  # Log individual entry errors

                if lines_consumed == 0:
                    print(
                        f"Warning: _parse_single_entry consumed 0 lines at index {i}. Incrementing by 1 to avoid loop.",
                        file=sys.stderr)
                    i += 1
                else:
                    i += lines_consumed
    else:
        print("Info: No sales data provided in clipboard_content to parse.")

    # --- Save to ML data ---
    # Save if we have API data OR if we successfully parsed some sales data.
    if api_data or processed_grid:
        try:
            machine_learning_save(processed_grid, discogs_release_id, api_data)
        except Exception as e:
            print(f"Error during machine_learning_save call: {e}", file=sys.stderr)
            status_message = f"{status_message or ''} Error saving ML data.".strip()
    else:
        print("Info: No API data and no processed sales data. Nothing to save for ML.", file=sys.stderr)

    return processed_grid, status_message


def machine_learning_save(processed_grid, discogs_release_id=None, api_data=None):
    """
    Saves or updates release information and its associated sales data to the ML data file.
    The ML data is structured as a dictionary keyed by discogs_release_id.
    Each release entry contains API metadata and a list of its sales history.
    Sales data is de-duplicated based on date, quality score, and native price.
    """
    all_releases_data = read_ml_data()

    if not isinstance(all_releases_data, dict):
        print("Warning: ML data file is not a dictionary. Initializing as empty. "
              "If you have old list-based data, it needs migration.", file=sys.stderr)
        all_releases_data = {}

    if not discogs_release_id:
        if processed_grid:  # Only warn if there was sales data that couldn't be associated
            print("Warning: Sales data provided but no discogs_release_id. "
                  "These sales will not be saved in the release-centric ML data structure.", file=sys.stderr)
        # If no ID and no sales, it's fine, just nothing to do.
        return

    release_key = str(discogs_release_id)
    new_sales_added_count = 0
    api_data_updated_flag = False

    release_entry = all_releases_data.get(release_key)

    if release_entry is None:
        release_entry = {}
        if api_data:
            release_entry.update(api_data)
            api_data_updated_flag = True  # API data added for new entry
        release_entry['sales_history'] = []
        all_releases_data[release_key] = release_entry
        print(f"Info: Creating new ML data entry for Release ID: {release_key}", file=sys.stderr)
    else:
        if api_data:
            # Check if API data has actually changed before updating and setting flag
            # This avoids unnecessary writes if API data is identical.
            # A simple way is to compare relevant fields or the whole dict if structure is fixed.
            # For now, assume if api_data is provided, it's intended for update.
            # A more sophisticated check:
            # if any(release_entry.get(k) != v for k, v in api_data.items()):
            #     release_entry.update(api_data)
            #     api_data_updated_flag = True
            #     print(f"Info: Updating API data for existing Release ID: {release_key}", file=sys.stderr)
            # else:
            #     print(f"Info: Provided API data for Release ID {release_key} is same as existing. No update to API fields.", file=sys.stderr)
            release_entry.update(api_data)  # Simpler: always update if api_data is present
            api_data_updated_flag = True
            print(f"Info: Updating/refreshing API data for existing Release ID: {release_key}", file=sys.stderr)

        if 'sales_history' not in release_entry or not isinstance(release_entry['sales_history'], list):
            release_entry['sales_history'] = []

    existing_sale_identifiers_for_release = set()
    for sale in release_entry['sales_history']:
        sale_identifier = (
            sale.get('date', ''),
            round(sale.get('quality', 0.0), 5),  # Round for consistent comparison
            str(sale.get('native_price', ''))  # Ensure native_price is string for consistency
        )
        existing_sale_identifiers_for_release.add(sale_identifier)

    current_batch_sale_identifiers = set()

    if processed_grid:
        for row in processed_grid:
            if len(row) >= 7:
                sale_date = row[0]
                native_price_from_row = str(row[4] or '')  # Ensure string, handle None
                quality_score = row[5]
                inflation_adjusted_price = row[3]
                sale_specific_comment = row[6]

                current_sale_de_dup_key = (
                    sale_date or '',
                    round(quality_score, 5),
                    native_price_from_row
                )

                if current_sale_de_dup_key in existing_sale_identifiers_for_release or \
                        current_sale_de_dup_key in current_batch_sale_identifiers:
                    continue

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

    # Determine if a write is needed: if new sales were added OR if API data was updated/added.
    if new_sales_added_count > 0 or api_data_updated_flag:
        write_ml_data(all_releases_data)
        artist_name_for_log = release_entry.get('api_artist', 'N/A')
        album_title_for_log = release_entry.get('api_title', 'N/A')
        log_message_parts = []
        if api_data_updated_flag:
            log_message_parts.append("API data processed/updated")
        if new_sales_added_count > 0:
            log_message_parts.append(f"added {new_sales_added_count} new sales")

        print(f"Info: For Release ID '{release_key}' ({artist_name_for_log} - {album_title_for_log}): "
              f"{' and '.join(log_message_parts)}. ML data file updated.", file=sys.stderr)
    else:
        artist_name_for_log = release_entry.get('api_artist', 'N/A') if release_entry else 'N/A'
        album_title_for_log = release_entry.get('api_title', 'N/A') if release_entry else 'N/A'
        print(f"Info: No new unique sales or API data updates for Release ID '{release_key}' "
              f"({artist_name_for_log} - {album_title_for_log}). ML data file not modified for this entry.",
              file=sys.stderr)


def points_match(grid_row, point_to_delete, tolerance=0.001):
    """
    Checks if a row from the processed grid matches a point selected for deletion.
    Compares date, quality score, and price. Comment matching is optional/flexible.
    """
    if len(grid_row) < 6:  # Needs at least date, price, score
        return False

    grid_score = grid_row[5]
    grid_price = grid_row[3]  # Inflation-adjusted price
    grid_date = grid_row[0]
    # grid_comment = grid_row[6] if len(grid_row) > 6 else "" # Comment matching can be tricky

    delete_score = point_to_delete.get('quality')
    delete_price = point_to_delete.get('price')  # This is the inflation-adjusted price from the chart
    delete_date = point_to_delete.get('date')
    # delete_comment = point_to_delete.get('comment', "")

    if delete_score is None or delete_price is None or delete_date is None:
        return False

    # Ensure types are compatible for comparison, especially for floats
    try:
        grid_score_float = float(grid_score)
        delete_score_float = float(delete_score)
        grid_price_float = float(grid_price) if grid_price is not None else -1  # Handle None grid_price
        delete_price_float = float(delete_price)
    except (ValueError, TypeError):
        return False  # Cannot compare if conversion fails

    score_match = math.isclose(grid_score_float, delete_score_float, rel_tol=tolerance)
    price_match = math.isclose(grid_price_float, delete_price_float,
                               rel_tol=tolerance) if grid_price is not None else False
    date_match = (grid_date == delete_date)
    # comment_match = ((grid_comment or "") == (delete_comment or "")) # Keep comment matching simple or remove

    return score_match and price_match and date_match  # and comment_match


def delete_points(points_to_delete_json, processed_grid):
    """
    Filters the processed grid, removing rows that match points marked for deletion.
    """
    points_to_delete = []
    if points_to_delete_json:
        try:
            points_to_delete = json.loads(points_to_delete_json)
        except json.JSONDecodeError:
            print("Warning: Could not decode points_to_delete JSON. Proceeding without deleting points.",
                  file=sys.stderr)
            points_to_delete = []

    deleted_count = 0

    if points_to_delete and processed_grid:
        initial_count = len(processed_grid)
        filtered_grid = []
        for row in processed_grid:
            should_delete = False
            for point in points_to_delete:
                if points_match(row, point):
                    should_delete = True
                    break
            if not should_delete:
                filtered_grid.append(row)

        deleted_count = initial_count - len(filtered_grid)
        if deleted_count > 0:
            print(f"Info: Deleted {deleted_count} point(s) based on selection.", file=sys.stderr)
        processed_grid = filtered_grid
    return processed_grid, deleted_count


def extract_tuples(processed_grid):
    # gets dates and comments from the processed_grid to go in the output
    dates = []
    comments = []
    qualities = []
    prices = []

    for row in processed_grid:
        dates.append(row[0])
        qualities.append(row[5])
        prices.append(row[3])  # Inflation-adjusted price
        comments.append(row[6] if len(row) > 6 and row[6] else "")
    return qualities, prices, dates, comments


def merge_and_deduplicate_grids(grid1, grid2):
    """
    Merges two grids (lists of lists/tuples) and removes duplicates.
    A row is identified by (date, quality_score_rounded, native_price_str).
    """
    seen_row_identifiers = set()
    merged_grid = []

    for grid in [grid1, grid2]:
        for row_list in grid:
            if len(row_list) >= 7:  # Ensure row has enough elements
                sale_date = row_list[0]
                quality_score = row_list[5]
                native_price = str(row_list[4] or '')  # Ensure string, handle None

                row_identifier = (
                    sale_date or '',
                    round(quality_score, 5),  # Round for consistency
                    native_price
                )
                if row_identifier not in seen_row_identifiers:
                    merged_grid.append(list(row_list))  # Store as list
                    seen_row_identifiers.add(row_identifier)
            else:
                # Handle rows with unexpected structure, perhaps log or skip
                print(f"Warning: Skipping row in merge_and_deduplicate_grids due to unexpected structure: {row_list}",
                      file=sys.stderr)
    return merged_grid


def manage_processed_grid(discogs_data, start_date, points_to_delete_json, add_data_str, discogs_release_id=None):
    """
    Manages the lifecycle of the processed sales data grid.
    Fetches API data if discogs_release_id is provided.
    Parses new sales data from discogs_data.
    Optionally merges with previously saved sales data.
    Handles deletion of points.
    Saves the final grid (for charting) and triggers ML data saving.

    Args:
        discogs_data (str): Raw Discogs sales history text.
        start_date (str): Start date for filtering sales ('YYYY-MM-DD').
        points_to_delete_json (str): JSON string of points to delete.
        add_data_str (str): 'True' to merge with saved data, 'False' otherwise.
        discogs_release_id (str, optional): The Discogs release ID.

    Returns:
        tuple: (final_grid, deleted_count, status_message_from_parsing)
               - final_grid (list): The final list of sales data tuples for charting.
               - deleted_count (int): Number of points deleted.
               - status_message_from_parsing (str or None): Status from initial parsing.
    Raises:
        ValueError: If no data points are available after processing.
    """
    current_grid_for_processing = []
    status_message_from_parsing = None
    api_data_for_release = None

    if discogs_release_id:
        print(f"Info: Managing grid for Discogs Release ID: {discogs_release_id}")
        api_data_for_release = fetch_api_data(discogs_release_id)
        if api_data_for_release:
            print(
                f"Info: API data fetched for: {api_data_for_release.get('api_artist', 'N/A')} - {api_data_for_release.get('api_title', 'N/A')}")
        else:
            print(f"Warning: Failed to fetch API data for Release ID: {discogs_release_id}.", file=sys.stderr)
    else:
        print("Info: No Discogs Release ID provided for grid management.")

    if discogs_data and discogs_data.strip():
        newly_parsed_data, status_message_from_parsing = make_processed_grid(
            discogs_data,
            start_date,
            discogs_release_id,
            api_data_for_release
        )
        # status_message_from_parsing from make_processed_grid is now more of a general status

        if add_data_str == "True":
            saved_grid_for_charting = read_save_value("processed_grid", [])  # This is the old chart-specific grid
            current_grid_for_processing = merge_and_deduplicate_grids(newly_parsed_data, saved_grid_for_charting)
            print(
                f"Info: Merged {len(newly_parsed_data)} new sales with {len(saved_grid_for_charting)} saved sales. Result: {len(current_grid_for_processing)} sales.")
        else:
            current_grid_for_processing = newly_parsed_data
            print(f"Info: Using {len(current_grid_for_processing)} newly parsed sales (add_data is False).")
    else:  # No new discogs_data pasted
        if add_data_str == "True":  # If adding, but no new data, load saved.
            current_grid_for_processing = read_save_value("processed_grid", [])
            print(
                f"Info: No new sales data pasted, add_data is True. Loaded {len(current_grid_for_processing)} sales from saved chart data.")
        else:  # No new data, and not adding, so grid starts empty for charting.
            current_grid_for_processing = []
            print("Info: No new sales data pasted, add_data is False. Charting grid starts empty.")

        # If no sales data pasted, but we have API data, we still need to trigger ML save
        if api_data_for_release and discogs_release_id:
            print("Info: No sales data pasted, but API data is available. Triggering ML save for API context.")
            _, status_from_api_only_save = make_processed_grid(
                "", start_date, discogs_release_id, api_data_for_release
            )
            if status_from_api_only_save and not status_message_from_parsing:
                status_message_from_parsing = status_from_api_only_save

    final_grid_for_charting, deleted_count = delete_points(points_to_delete_json, current_grid_for_processing)

    if not final_grid_for_charting:
        # Allow empty grid if it's an API-only call (no sales data pasted, and not adding from save)
        # OR if all points were deleted.
        is_api_only_call_no_sales_pasted = not (discogs_data and discogs_data.strip()) and not (add_data_str == "True")

        if not is_api_only_call_no_sales_pasted and deleted_count != len(current_grid_for_processing):
            # If it's not an API-only call AND not all points were deleted to make it empty, then it's an issue.
            # The original status_message_from_parsing might be relevant if parsing failed.
            # If parsing was okay but deletion made it empty, that's also a state.
            # For now, if it's empty and wasn't meant to be, raise error.
            # This helps prevent charting an empty graph unless intended.
            if status_message_from_parsing:
                raise ValueError(f"No data for analysis. Parsing status: {status_message_from_parsing}")
            raise ValueError("No data points available for analysis after processing, loading, or deletion.")
        elif final_grid_for_charting:  # Should not happen if not final_grid_for_charting is true
            pass  # This case is contradictory
        else:  # Grid is empty, but it's an API-only call or all points deleted. This is acceptable.
            print(
                "Info: Final grid for charting is empty. This may be due to an API-only data update or all points being deleted.")

    write_save_value(final_grid_for_charting, "processed_grid")  # Saves the grid used for charting


    return final_grid_for_charting, deleted_count, status_message_from_parsing, api_data_for_release


def fetch_api_data(release_id):
    """
    Fetches detailed information for a given release ID from the Discogs API,
    including the original year from its master release.

    Args:
        release_id (str or int): The Discogs release ID.

    Returns:
        dict: A dictionary containing extracted release information (e.g., api_year,
              api_artist, api_title, api_genres, api_styles, api_country, api_label,
              api_catno, api_format_descriptions, api_notes, api_community_have,
              api_community_want, api_community_rating_average, api_community_rating_count,
              api_master_id, api_original_year),
              or None if an error occurs or ID is not found.
    """
    if not release_id:
        return None

    try:
        d = discogs_client.Client('VinylPriceCalculator/0.1', user_token=DISCOGS_USER_TOKEN)
        release = d.release(int(release_id))
        api_data = {}

        api_data['api_year'] = getattr(release, 'year', None)
        api_data['api_title'] = getattr(release, 'title', None)
        api_data['api_artist'] = getattr(release.artists[0], 'name', None) if release.artists else None
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
        api_data['api_format_descriptions'] = list(set(format_descriptions))  # Deduplicate

        api_data['api_notes'] = release.data.get('notes', None)  # Notes often in .data

        community_data = release.data.get('community', {})
        api_data['api_community_have'] = community_data.get('have', None)
        api_data['api_community_want'] = community_data.get('want', None)
        rating_data = community_data.get('rating', {})
        api_data['api_community_rating_average'] = rating_data.get('average', None)
        api_data['api_community_rating_count'] = rating_data.get('count', None)

        master_id = getattr(release, 'master_id', None)
        if not master_id or master_id == 0:  # Check .data if not on object or is 0
            master_id_from_data = release.data.get('master_id')
            if master_id_from_data and master_id_from_data != 0:
                master_id = master_id_from_data
            else:
                master_id = None

        api_data['api_master_id'] = master_id
        api_data['api_original_year'] = None

        if master_id:
            print(
                f"Info: Found Master ID: {master_id} for Release ID: {release_id}. Fetching master release details...")
            try:
                master_release = d.master(int(master_id))
                master_release.refresh()  # Crucial step to get all data for the master release
                print(f"Info: Refresh complete for Master ID: {master_id}.")

                original_year_from_attr = getattr(master_release, 'year', None)
                original_year_from_data = master_release.data.get('year')

                if original_year_from_data and original_year_from_data != 0:
                    api_data['api_original_year'] = original_year_from_data
                elif original_year_from_attr and original_year_from_attr != 0:  # Fallback to attribute if .data failed
                    api_data['api_original_year'] = original_year_from_attr

                if api_data['api_original_year']:
                    print(
                        f"Info: Successfully fetched original year: {api_data['api_original_year']} for Master ID: {master_id}")
                elif original_year_from_data == 0 or original_year_from_attr == 0:  # Explicitly 0
                    print(f"Warning: Master ID {master_id} (Release ID: {release_id}) has 'year' as 0.",
                          file=sys.stderr)
                else:  # Both were None
                    print(
                        f"Warning: Master ID {master_id} (Release ID: {release_id}) found, but 'year' is missing or None after refresh.",
                        file=sys.stderr)

            except discogs_client.exceptions.HTTPError as master_http_err:
                if master_http_err.status_code == 404:
                    print(
                        f"Warning: Discogs API Error - Master ID {master_id} (for Release ID: {release_id}) not found (404).",
                        file=sys.stderr)
                else:
                    print(
                        f"Warning: Discogs API HTTP Error for Master ID {master_id} (for Release ID: {release_id}): {master_http_err}",
                        file=sys.stderr)
            except Exception as master_e:
                print(
                    f"Warning: An unexpected error occurred while fetching Master ID {master_id} (for Release ID: {release_id}): {master_e}",
                    file=sys.stderr)
        else:
            print(f"Info: No valid Master ID found on Release ID: {release_id}.")

        print(f"Info: Successfully processed API data for release ID: {release_id}")
        return api_data

    except discogs_client.exceptions.HTTPError as http_err:
        if http_err.status_code == 404:
            print(f"Warning: Discogs API Error - Release ID {release_id} not found (404).", file=sys.stderr)
        elif http_err.status_code == 401:
            print(f"Error: Discogs API Error - Unauthorized (401). Check your User Token.", file=sys.stderr)
        else:
            print(f"Warning: Discogs API HTTP Error for Release ID {release_id}: {http_err}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred while fetching API data for Release ID {release_id}: {e}",
              file=sys.stderr)
        return None
