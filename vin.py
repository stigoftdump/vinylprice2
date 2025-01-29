import pyperclip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sys
from datetime import datetime
import tkinter as tk
from tkinter import messagebox


#TO do list
# solve the data being weird problem
# Make the screen look nicer
# Error checking.

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

# gets the inputs from the sys arguments
if len(sys.argv) < 4:
    print("Error: data is missing.")
    sys.exit(1)
try:
    reqscore = float(sys.argv[1])  # Read the first argument as reqscore
    shop_var = float(sys.argv[2])  # Read the second argument as shop_var
    start_date = sys.argv[3] # read the third argument as start_date
except ValueError:
    print("Error: Both reqscore and shop_var must be numbers.")
    sys.exit(1)

# Get clipboard content
clipboard_content = pyperclip.paste()

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
                # Remove '£' and convert to a float
                filtered_grid[i] = filtered_grid[i][:3] + (float(price_str[1:]),)
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

qualities = []
prices = []

for row in processed_grid:
    if len(row) >= 5:  # Ensure there are at least 5 elements
        qualities.append(row[4])  # Quality column (score)
        prices.append(row[3])  # Price column

# Convert to numpy arrays for regression
X = np.array(qualities).reshape(-1, 1)  # Quality as independent variable
y = np.array(prices)  # Price as dependent variable

# Apply polynomial regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# # Train the model
model = LinearRegression()
model.fit(X_poly, y)

# Get the predicted values
y_pred = model.predict(X_poly)

# Calculate absolute percentage errors
percentage_errors = np.abs((y - y_pred) / y) * 100

# Calculates the root mean squared error - this is the +/- of the dataset
mse = mean_squared_error(y, model.predict(X_poly))
rmse = np.sqrt(mse)

# Predict price for a new quality (e.g., quality = 10)
new_quality = reqscore
new_quality_poly = poly.transform([[new_quality]])
predicted_price = model.predict(new_quality_poly)[0]

#' new definition of upper bound'
upper_bound = predicted_price+rmse

# amends the predicated price to allow for the rmse
adjusted_price = predicted_price + (rmse*shop_var)

# gets the actual price
actual_price = realprice(adjusted_price)

# Generate smooth data points for the regression line
quality_range = np.linspace(min(qualities), max(qualities), 100).reshape(-1, 1)  # Smooth range of qualities
quality_range_poly = poly.transform(quality_range)  # Transform the range for polynomial features

# Predict prices for the smooth quality range
predicted_prices = model.predict(quality_range_poly)

# Plot the scatter and the regression line
plt.scatter(qualities, prices, color='red')  # Scatter plot of actual data
plt.plot(quality_range, predicted_prices, color='blue', linestyle='-')  # Polynomial regression line
plt.axhline(y=actual_price, color='green', linestyle='--', xmin=0, xmax=(float(reqscore) - min(qualities)) / (max(qualities) - min(qualities)))
plt.axhline(y=predicted_price, color='orange', linestyle='--', xmin=0, xmax=(float(reqscore) - min(qualities)) / (max(qualities) - min(qualities)))
plt.axhline(y=upper_bound, color='purple', linestyle='--', xmin=0, xmax=(float(reqscore) - min(qualities)) / (max(qualities) - min(qualities)))
plt.axvline(x=float(reqscore), color='green', linestyle='--', ymin=0, ymax=(actual_price - min(prices)) / (max(prices) - min(prices)))
plt.axvline(x=float(reqscore), color='purple', linestyle='--', ymin=(actual_price - min(prices)) / (max(prices) - min(prices)), ymax=(upper_bound - min(prices)) / (max(prices) - min(prices)))
plt.scatter(float(reqscore), actual_price, color='green', s=100) # user point
plt.scatter(float(reqscore), predicted_price, color='orange', s=100) # user point
plt.scatter(float(reqscore), upper_bound, color='purple', s=100) # user point
plt.title('Polynomial Regression: Quality vs Price')
plt.xlabel('Quality')
plt.ylabel('Price (£)')
plt.savefig('static/images/chart.png')  # Save as PNG

# output for sending to flask
print(f"{round(predicted_price,2)},{round(predicted_price+rmse,2)},{round(actual_price,2)}")