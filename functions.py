import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np
import datetime
from datetime import datetime
import sys
import matplotlib.pyplot as plt

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

# UK inflation
ANNUAL_CPI_DATA_1974_BASE = {
    "2000": 671.8,
    "2001": 683.7,
    "2002": 695.1,
    "2003": 715.2,
    "2004": 736.5,
    "2005": 757.3,
    "2006": 781.5,
    "2007": 815.0,
    "2008": 847.5,
    "2009": 843.0,
    "2010": 881.9,
    "2011": 927.8,
    "2012": 957.6,
    "2013": 986.7,
    "2014": 1010.0,
    "2015": 1020.0,
    "2016": 1037.7,
    "2017": 1074.9,
    "2018": 1110.8,
    "2019": 1139.3,
    "2020": 1156.4,
    "2021": 1203.2,
    "2022": 1342.6,
    "2023": 1472.7,
    "2024": 1525.5, # Latest data point in this set
}



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

def show_error_message(message):
    root = tk.Tk()  # Initialize Tkinter root
    root.withdraw()  # Hide the root window
    messagebox.showerror("Clipboard Error", message)  # Show the error message
    root.destroy()  # Destroy the root window after the message is closed

# converts the record quality and sleeve quality into a score'
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

def sigmoid_plus_exponential(x, a1, b1, c1, a_exp, b_exp, c_exp, d):
    """
    Combines a sigmoid function with an exponential increase, plus a constant.
    - a1, b1, c1: Parameters for the initial sigmoid rise
    - a_exp, b_exp, c_exp: Parameters for the exponential component
    - d: Base level offset
    """
    x = np.asarray(x)
    first_rise = a1 / (1 + np.exp(-b1 * (x - c1)))
    exponential_increase = a_exp * np.exp(b_exp * (x - c_exp))
    return first_rise + exponential_increase + d

# creates the processed grid data from clipboard data
def make_processed_grid(clipboard_content, start_date):

    # Check for the presence of "Order Date" and "Change Currency" in the clipboard content
    if "Order Date" not in clipboard_content or "Change Currency" not in clipboard_content:
        # Show error message box
        show_error_message("No Discogs data in clipboard. Go to the Discogs Sales History, CTRL-A to select all, CTRL-C to copy, then come back and re-run")
        sys.exit(0)  # Exit the script

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
    return processed_grid

# writes the chart
def plot_chart(qualities, prices, quality_range, predicted_prices, reqscore, predicted_price, upper_bound, actual_price):
    # Plot the scatter points and curve
    plt.scatter(qualities, prices, color='red', label='Items Sold')
    plt.plot(quality_range, predicted_prices, color='blue', linestyle='-', label='Best Fit')

    # Set y-axis limits
    plt.ylim(0, max(prices) * 1.1)

    # Set x-axis limits to the quality range
    plt.xlim(min(min(qualities), reqscore) * 0.97, max(max(qualities), reqscore) * 1.03)

    # Add grid for easier reading
    plt.grid(True, linestyle='--', alpha=0.7)

    # Horizontal reference lines
    plt.axhline(y=predicted_price, color='orange', linestyle='--',
                xmin=0, xmax=(float(reqscore) - 1) / 8,  # Adjusted for 1-9 range
                label='Predicted Price')
    plt.axhline(y=upper_bound, color='purple', linestyle='--',
                xmin=0, xmax=(float(reqscore) - 1) / 8,  # Adjusted for 1-9 range
                label='Upper Bound (RMSE)')
    plt.axhline(y=actual_price, color='green', linestyle='--',
                xmin=0, xmax=(float(reqscore) - 1) / 8,  # Adjusted for 1-9 range
                label='Actual Price')

    # Vertical reference line
    plt.axvline(x=float(reqscore), color='green', linestyle='--',
                ymin=0, ymax=actual_price / (max(prices) * 1.1))
    plt.axvline(x=float(reqscore), color='purple', linestyle='--',
                ymin=actual_price / (max(prices) * 1.1),
                ymax=upper_bound / (max(prices) * 1.1))

    # Mark specific points
    plt.scatter(float(reqscore), actual_price, color='green', s=100)
    plt.scatter(float(reqscore), predicted_price, color='orange', s=100)
    plt.scatter(float(reqscore), upper_bound, color='purple', s=100)

    # Add title and labels
    plt.title('Predicted Prices')
    plt.xlabel('Quality')
    plt.ylabel('Price (£)')
    plt.legend()

    # Create a reverse mapping from number to quality text
    number_to_quality = {v: k for k, v in quality_to_number.items()}

    # Gets labels size
    min_quality_label = int(min(min(qualities), reqscore) * 0.97)
    max_quality_label = int(max(max(qualities), reqscore) * 1.03)

    # Generate a list of all whole numbers within this range
    all_quality_numbers = range(min_quality_label, max_quality_label + 1)

    # Generate the corresponding text labels for all whole numbers
    all_quality_labels = [number_to_quality.get(q, str(q)) for q in all_quality_numbers]

    # Set the x-axis ticks and labels
    plt.xticks(all_quality_numbers, all_quality_labels, rotation=15, ha='right')
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping

    plt.savefig('static/chart.png')  # Save as PNG

def return_variables(argument):
    # returns the variables from the argument sent in sys
    if len(argument) < 4:
        print("Error: data is missing.")
        sys.exit(1)
    try:
        reqscore = float(argument[1])  # Read the first argument as reqscore
        shop_var = float(argument[2])  # Read the second argument as shop_var
        start_date = argument[3] # read the third argument as start_date
        add_data = argument[4] # gets the add_data flag
        max_price = argument[5] # gets the max_price
    except ValueError:
        print("Error: Both reqscore and shop_var must be numbers.")
        sys.exit(1)

    return reqscore, shop_var, start_date, add_data, max_price

def adjust_price_for_uk_inflation_annual(processed_grid):
    """
    Adjusts a price within a tuple based on ANNUAL UK inflation index data
    (1974=100 base) to its equivalent value relative to the latest year
    available in the dataset (2024).

    Args:
        processed_grid (tuple): A tuple where the first element (index 0) is a
                               date string ('YYYY-MM-DD') and the fourth element
                               (index 3) is the original price (numeric type like
                               int or float).

    Returns:
        tuple: A new tuple with the price at index 3 adjusted for inflation,
               rounded to 2 decimal places. Returns the original tuple if
               the date is invalid, falls outside the data range (2000-2024),
               price is invalid, or CPI data for the original year is missing
               (though the provided dict is complete for 2000-2024).
        None: Returns None if the input tuple structure is incorrect (e.g.,
              less than 4 elements).

    Notes:
        - Uses the specific annual index data provided (2000-2024, 1974=100).
        - Adjustment is based on annual average index values.
        - The adjustment reflects the equivalent value as of the average price
          level in the year LATEST_CPI_YEAR_STR (currently 2024), not today's date.
    """
    try:
        # Ensure tuple has enough elements before accessing them
        if len(processed_grid) < 4:
             raise IndexError("Input tuple requires at least 4 elements.")

        LATEST_CPI_YEAR_STR="2024"
        LATEST_CPI_VALUE = 1525.5
        date_str = processed_grid[0]
        original_price = processed_grid[3]
        # We don't strictly need today's date for calculation here, but can keep it for validation context
        today = datetime.date.today()

        # --- Validate and parse date ---
        try:
            original_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            original_year = original_date.year # Extract the year
        except ValueError:
            print(f"Warning: Invalid date format '{date_str}'. Expected YYYY-MM-DD. Returning original tuple.")
            return processed_grid
        except TypeError:
             print(f"Warning: Date element is not a string ('{date_str}'). Returning original tuple.")
             return processed_grid

        # --- Validate price ---
        try:
            if not isinstance(original_price, (int, float)):
                 raise TypeError("Price must be a numeric type (int or float).")
            original_price = float(original_price)
            if original_price < 0:
                print(f"Warning: Original price {original_price} is negative. Proceeding with calculation.")
        except (TypeError, ValueError) as e:
            print(f"Warning: Invalid price '{original_price}' ({e}). Returning original tuple.")
            return processed_grid

        # --- Check if original year is within the data range ---
        original_year_str = str(original_year)

        # --- CPI Lookup ---
        cpi_original = ANNUAL_CPI_DATA_1974_BASE[original_year_str]
        cpi_latest = LATEST_CPI_VALUE # Use the defined latest value from the dataset

        # Basic check for valid CPI values
        if cpi_original <= 0:
             print(f"Warning: Invalid historical CPI value ({cpi_original}) found for {original_year_str}. Returning original tuple.")
             return processed_grid

        # --- Inflation Calculation ---
        inflation_multiplier = cpi_latest / cpi_original
        adjusted_price = round(original_price * inflation_multiplier, 2)

        # --- Create the new tuple ---
        grid_list = list(processed_grid)
        grid_list[3] = adjusted_price
        new_processed_grid = tuple(grid_list)

        return new_processed_grid

    except IndexError:
        print(f"Error: Input tuple structure is incorrect or has too few elements. Expected at least 4 elements.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return processed_grid