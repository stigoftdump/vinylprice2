import pickle
import re
from datetime import datetime

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

def delete_above_max_price(processed_grid, max_price):
    # deletes above the max price
    processed_grid = [
        row for row in processed_grid
        if len(row) > 3 and isinstance(row[3], (int, float)) and row[3] < max_price
    ]
    return processed_grid

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

                    # Ensure enough parts for basic processing (Date, Q1, Q2, Price)
                    if len(data_parts) < 4:
                        print(f"Warning: Skipping malformed data row (too few columns): {row}")
                        i += 1
                        continue

                    # --- 1. Date Filtering ---
                    row_date_str = data_parts[0].strip()  # Get date string
                    current_date = datetime.strptime(row_date_str, '%Y-%m-%d').date()

                    if current_date >= start_date_obj:
                        # Convert tuple to list for modification
                        current_row_list = list(data_parts)

                        # --- 2. Price Conversion ---
                        # Check index 3 exists and is a string starting with '£'
                        if len(current_row_list) > 3 and isinstance(current_row_list[3], str) and current_row_list[
                            3].startswith('£'):
                            price_str = current_row_list[3]
                            try:
                                # Remove '£' and commas, then convert to a float
                                clean_price = price_str[1:].replace(',', '')
                                current_row_list[3] = float(clean_price)
                            except ValueError:
                                # Keep original string if conversion fails, maybe log it
                                print(f"Warning: Could not convert price '{price_str}' to number in row: {row}")
                                pass  # Keep original price string

                        # --- 3. Score Calculation ---
                        score = None  # Default score if calculation fails or not enough data
                        # Check indices 1 and 2 exist
                        if len(current_row_list) > 2:
                            record_quality = current_row_list[1]
                            sleeve_quality = current_row_list[2]
                            score = calculate_score(record_quality, sleeve_quality)
                        # Append score to the end of the *current* data elements
                        current_row_list.append(score)

                        # --- 4. Check for Comment on NEXT line ---
                        comment = None
                        # Check if there *is* a next line
                        if i + 1 < len(relevant_rows):
                            next_row = relevant_rows[i + 1]
                            # Check if it starts specifically with TAB then "Comments:"
                            if next_row.startswith('\tComments:'):
                                # Extract the comment text
                                comment_text = next_row.strip()  # Strip whitespace around comment line
                                # Remove "Comments:" prefix and strip again
                                comment = comment_text[len("Comments:"):].strip()
                                # Append the extracted comment to the row list
                                current_row_list.append(comment)
                                # We've processed the comment line, so skip it in the next iteration
                                i += 1

                        # Add the fully processed row (with potential score and comment)
                        intermediate_grid.append(current_row_list)

                except ValueError as ve:
                    # Handle potential errors during date parsing or float conversion within the loop
                    print(f"Warning: Skipping row due to data conversion error: {row} | Error: {ve}")
                except IndexError as ie:
                    # Handle potential errors if a row has fewer columns than expected
                    print(f"Warning: Skipping row due to missing columns: {row} | Error: {ie}")

            # Move to the next line (or the line after the comment if one was processed)
            i += 1

        # Convert the list of lists back to a list of tuples for the final output
        processed_grid_final = [tuple(item) for item in intermediate_grid]

    return processed_grid_final, status_message
