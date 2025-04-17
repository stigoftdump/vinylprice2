import pyperclip
import sys
from functions import *
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

def main():
    # gets the inputs from the sys arguments
    reqscore, shop_var, start_date, add_data, max_price = return_variables(sys.argv)

    # Ensure max_price is either an integer or None
    if max_price and max_price.isdigit():  # Check if it's not empty and is a number
        max_price = int(max_price)
    else:
        max_price = None  # Set to None explicitly if it's empty or invalid

    # Gets the processed_grid from the clipboard
    processed_grid = make_processed_grid(pyperclip.paste(), start_date)

    # adds in the maxprice and deletes if anything is above it.
    if max_price is not None:
        processed_grid = [
            row for row in processed_grid
            if len(row) > 3 and isinstance(row[3], (int, float))  and row[3] < max_price
        ]

    # If add_data is True, load the previously saved processed_grid and add it to the current one
    if add_data == "True":
        saved_processed_grid = load_processed_grid()
        processed_grid.extend(saved_processed_grid)

    # Save processed grid to file
    save_processed_grid(processed_grid)

    # graph logic to get the variables for the output
    qualities, prices, X_smooth, y_smooth_pred, reqscore, predicted_price, upper_bound, actual_price, upper_bound = graph_logic(reqscore, shop_var, processed_grid)

    # calls the function to plot the chart and saves it
    plot_chart(qualities, prices, X_smooth, y_smooth_pred, reqscore, predicted_price, upper_bound, actual_price)

    # output for sending to flask
    print(f"{round(predicted_price,2)},{round(upper_bound,2)},{round(actual_price,2)}")

if __name__ == "__main__":
    main()