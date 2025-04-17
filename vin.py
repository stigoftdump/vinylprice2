import functions
import sys
import json

def main():
    status_message = None

    # gets the inputs from the sys arguments
    reqscore, shop_var, start_date, add_data, max_price, discogs_data = functions.return_variables(sys.argv)

    # Ensure max_price is either an integer or None
    if max_price and max_price.isdigit():  # Check if it's not empty and is a number
        max_price = int(max_price)
    else:
        max_price = None  # Set to None explicitly if it's empty or invalid

    # Gets the processed_grid from the clipboard
    processed_grid, status_message = functions.make_processed_grid(discogs_data, start_date)
    # if there was no discogs data, return the status_message and exit
    if status_message is not None:
        print(f"0,0,0,{status_message}")
        sys.exit(0)
    else:
        # adds in the maxprice and deletes if anything is above it.
        if max_price is not None:
            processed_grid = [
                row for row in processed_grid
                if len(row) > 3 and isinstance(row[3], (int, float))  and row[3] < max_price
            ]

        # If add_data is True, load the previously saved processed_grid and add it to the current one
        if add_data == "True":
            saved_processed_grid = functions.load_processed_grid()
            processed_grid.extend(saved_processed_grid)

        # Save processed grid to file
        functions.save_processed_grid(processed_grid)

        # graph logic to get the variables for the output
        qualities, prices, X_smooth, y_smooth_pred, reqscore, predicted_price, upper_bound, actual_price, upper_bound = functions.graph_logic(reqscore, shop_var, processed_grid)

        # Create chart data in JSON format
        chart_data = {
            "labels": [str(q) for q in qualities],  # Convert qualities to strings for labels
            "prices": prices,
            "predicted_prices": list(y_smooth_pred),  # Convert numpy array to list
            "predicted_qualities": list(X_smooth),  # Add predicted qualities
            "reqscore": reqscore,
            "predicted_price": predicted_price,
            "upper_bound": upper_bound,
            "actual_price": actual_price
        }

        # make status "Okay" if there are no problems
        if status_message is None:
            status_message = "Completed"

        # output for sending to flask
        output_data = {
            "calculated_price": round(predicted_price, 2),
            "upper_bound": round(upper_bound, 2),
            "actual_price": round(actual_price, 2),
            "status_message": status_message,
            "chart_data": chart_data  # Include chart data in the output
        }

        print(json.dumps(output_data))  # Print the JSON string

if __name__ == "__main__":
    main()