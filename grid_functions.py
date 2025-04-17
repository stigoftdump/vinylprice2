import pickle
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

        # Split each row into cells based on tabs ('\t') and convert to tuples
        grid = [tuple(row.split('\t')) for row in rows]

        # Exclude header row by skipping the first row (assuming it's the header)
        # also removes any purchases from before the start date
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
        filtered_grid = [
            row for row in grid[1:]
            if row[0].strip() and datetime.strptime(row[0].strip(), '%Y-%m-%d').date() >= start_date_obj
            ]

        # Convert the fourth element (index 3) from a string with '£' to a number
        for i in range(len(filtered_grid)):
            if len(filtered_grid[i]) > 3:  # Ensure there is a fourth element
                price_str = filtered_grid[i][3]
                if price_str.startswith('£'):  # Check if the string starts with '£'
                    try:
                        # Remove '£' and commas, then convert to a float
                        clean_price = price_str[1:].replace(',', '')
                        filtered_grid[i] = filtered_grid[i][:3] + (float(clean_price),)
                    except ValueError:
                        print(f"Error converting {price_str} to a number.")

        # Process each row to calculate the score for the record and sleeve qualities
        processed_grid = []
        for row in filtered_grid:
            if len(row) > 2:  # Ensure there are enough elements (record and sleeve qualities)
                record_quality = row[1]  # Second element (record quality)
                sleeve_quality = row[2]  # Third element (sleeve quality)
                score = calculate_score(record_quality, sleeve_quality)
                # Add the score as the last element in the tuple
                processed_grid.append(row + (score,))
            else:
                # In case there are rows with missing data
                processed_grid.append(row + (None,))

    return processed_grid, status_message
