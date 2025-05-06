from flask import Flask, render_template, request, session, send_from_directory
import json
import webbrowser
import threading
import time
from datetime import datetime
# We will call graph_logic and grid_functions directly for more control over the grid
from grid_functions import load_processed_grid, save_processed_grid, delete_points, make_processed_grid
from functions import graph_logic # Import graph_logic
import os

app = Flask(__name__)
# Add a secret key for session management
# IMPORTANT: Replace 'your_secret_key_here_securely' with a real,
# complex and unique secret key in your actual application.
app.secret_key = 'your_secret_key_here_securely'

# --- Keep existing shop_var functions ---
# Load shop_var from a file
def read_shop_var():
    """
    Reads the 'shop_var' value from the save file.
    """
    SAVE_FILE = "save_data.json"
    try:
        with open(SAVE_FILE, 'r') as f:
            data = json.load(f)
            return data.get("shop_var", 0.8)  # Default to 0.8 if not set
    except (FileNotFoundError, json.JSONDecodeError):
        return 0.8  # Default to 0.8

# Write shop_var to a file
def write_shop_var(value):
    """
    Writes the given 'shop_var' value to the save file.
    """
    SAVE_FILE = "save_data.json"
    try:
        with open(SAVE_FILE, 'w') as f:
            json.dump({"shop_var": value}, f)
    except IOError as e:
        print(f"Error writing to {SAVE_FILE}: {e}")


# Add a shutdown route
@app.route('/shutdown', methods=['POST'])
def shutdown():
    """
    Handles the shutdown signal sent from the browser's 'beforeunload' event
    to terminate the Flask development server process.
    """
    # Note: This shutdown mechanism is for development/demonstration.
    # Proper shutdown in production requires a different approach.
    os._exit(0)

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Handles GET and POST requests for the main page.

    GET: Renders the main page with default or saved values.
    POST: Processes the form submission, calculates the price,
           and renders the page with the results and updated form values.

    Returns:
        str: Rendered HTML template ('index.html') with context variables.
    """
    calculated_price = None
    adjusted_price = None
    actual_price = None
    status_message = ""
    chart_data = {}  # Initialize chart_data
    # processed_grid will be determined within the POST block

    # Variables to hold form values for rendering (will be populated below)
    # Load initial values for GET request or use defaults
    # On GET, default values are used. On POST, submitted values are used.
    pasted_discogs_data_display = "" # Default empty
    media_display = request.form.get("media", 6) # Default 6 if not in form
    sleeve_display = request.form.get("sleeve", 6) # Default 6 if not in form
    shop_var_display = request.form.get("shop_var", read_shop_var()) # Load from file or form
    start_date_display = request.form.get("start_Date", "2020-01-01") # Default
    add_data_display = request.form.get("add_data", "off") == "on" # Default False


    if request.method == "POST":
        # --- Read form values ---
        # Always read from form on POST to get latest user input
        media = int(request.form.get("media", 6))
        sleeve = int(request.form.get("sleeve", 6))
        shop_var = float(request.form.get("shop_var", 0.8))
        start_date = request.form.get("start_Date", "2020-01-01")
        add_data_flag = request.form.get("add_data", "off") == "on"
        points_to_delete_json = request.form.get('selected_points_to_delete', '[]')
        discogs_data = request.form.get('pasted_discogs_data', '') # Always get raw data from form

        # Update display variables with submitted values for rendering
        media_display = media
        sleeve_display = sleeve
        shop_var_display = shop_var
        start_date_display = start_date
        add_data_display = add_data_flag

        # Save the shop_var value for next session
        write_shop_var(shop_var)

        # Validate date format briefly
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
        except ValueError:
             status_message = "Invalid start date format. Please use YYYY-MM-DD."
             # Keep the discogs data in the textarea for rendering
             pasted_discogs_data_display = discogs_data
             return render_template("index.html",
                       pasted_discogs_data=pasted_discogs_data_display,
                       media=media_display,
                       sleeve=sleeve_display,
                       shop_var=shop_var_display,
                       start_date=start_date_display,
                       add_data=add_data_display,
                       status_message=status_message,
                       calculated_price=None, adjusted_price=None, actual_price=None, chart_data={})

        # --- Determine the base data for the current calculation ---
        current_calc_grid = [] # Initialize the grid for this calculation

        if discogs_data.strip(): # If new Discogs data is pasted
            print("New Discogs data pasted. Parsing.")
            parsed_new_grid, status_message = make_processed_grid(discogs_data, start_date)
            if parsed_new_grid:
                 current_calc_grid = parsed_new_grid
                 # When new data is pasted, reset the session grid to this new data
                 session['processed_grid'] = current_calc_grid
            else:
                 # If parsing new data failed, clear session and return error
                 if 'processed_grid' in session:
                      del session['processed_grid']
                 return render_template("index.html",
                       pasted_discogs_data=discogs_data,
                       media=media_display,
                       sleeve=sleeve_display,
                       shop_var=shop_var_display,
                       start_date=start_date_display,
                       add_data=add_data_display,
                       status_message=status_message,
                       calculated_price=None, adjusted_price=None, actual_price=None, chart_data={})

        elif 'processed_grid' in session: # If no new data, try to load from session
            print("No new data pasted. Loading processed_grid from session.")
            current_calc_grid = session['processed_grid']
            # Note: start_date filtering was applied during initial make_processed_grid.
            # If start_date changes on a rerun without new data, the session grid isn't
            # re-filtered by date. This is a current limitation based on the request.
        else:
             # No new data and no session data
             print("No new data pasted and no processed_grid in session.")
             status_message = "No data available to process. Please paste Discogs data."
             # Keep data in textarea for rendering
             pasted_discogs_data_display = discogs_data
             return render_template("index.html",
                       pasted_discogs_data=pasted_discogs_data_display,
                       media=media_display,
                       sleeve=sleeve_display,
                       shop_var=shop_var_display,
                       start_date=start_date_display,
                       add_data=add_data_display,
                       status_message=status_message,
                       calculated_price=None, adjusted_price=None, actual_price=None, chart_data={})

        # --- Handle "Add Data" and Merge ---
        # The grid for graph calculation starts with the current_calc_grid
        processed_grid_for_graph = current_calc_grid.copy() # Work on a copy

        if add_data_flag:
            print("Add to Previous Data checked. Loading and merging saved grid.")
            saved_processed_grid = load_processed_grid() # Load from pickle
            # Extend the grid for calculation with the saved data
            processed_grid_for_graph.extend(saved_processed_grid)
            # Note: De-duplication might be needed here if merging the same data sources is a concern.

        # --- Apply Deletions from the Current Request ---
        # Apply the points selected for deletion in THIS submission to the
        # processed_grid_for_graph (which is either base or merged).
        initial_grid_size_before_current_deletions = len(processed_grid_for_graph)
        processed_grid_for_graph, deleted_count_current_request = delete_points(points_to_delete_json, processed_grid_for_graph)

        # --- Check if grid is empty after all processing ---
        if not processed_grid_for_graph:
             status_message = "No data points available for analysis after filtering/deletion."
             pasted_discogs_data_display = discogs_data # Keep data in textarea
             return render_template("index.html",
                       pasted_discogs_data=pasted_discogs_data_display,
                       media=media_display,
                       sleeve=sleeve_display,
                       shop_var=shop_var_display,
                       start_date=start_date_display,
                       add_data=add_data_display,
                       status_message=status_message,
                       calculated_price=None, adjusted_price=None, actual_price=None, chart_data={})

        # --- Save the final processed_grid to session AND pickle ---
        # Save the grid used for the graph calculation (including merges and current deletions)
        # to the session for the next calculation in the current session.
        session['processed_grid'] = processed_grid_for_graph
        # Save the grid used for the graph calculation to pickle for the 'Add Data'
        # feature in future sessions.
        save_processed_grid(processed_grid_for_graph)

        # Calculate quality from media and sleeve for graph_logic
        quality = media - ((media - sleeve) / 3)

        # --- Call graph_logic with the final processed_grid ---
        try:
            # graph_logic takes the reqscore (calculated quality), shop_var, and the processed_grid.
            qualities_list, prices_list, X_smooth, y_smooth_pred, predicted_price, upper_bound, actual_price = graph_logic(quality, shop_var, processed_grid_for_graph)

            # Re-assemble chart data based on the final processed_grid and graph_logic results
            # Extract dates and comments from the final processed_grid
            dates = [row[0] if len(row) > 0 else None for row in processed_grid_for_graph]
            # Ensure comment extraction handles potential None or index errors
            comments = [row[6] if len(row) > 6 and row[6] is not None else "" for row in processed_grid_for_graph]

            # Extract quality scores and prices directly from the processed_grid_for_graph
            # These should align with qualities_list and prices_list from graph_logic if
            # graph_logic simply extracts them, but using the grid directly for chart data source
            # ensures they match the data points actually used for plotting.
            grid_qualities = [row[5] for row in processed_grid_for_graph]
            grid_prices = [row[3] for row in processed_grid_for_graph]


            chart_data = {
                "labels": [str(q) for q in grid_qualities],  # Use scores from the final grid (converted to string for labels)
                "prices": grid_prices,                     # Use prices from the final grid
                "predicted_prices": list(y_smooth_pred),
                "predicted_qualities": list(X_smooth),
                "reqscore": quality, # Use the calculated quality for the requested point
                "dates": dates, # Dates from the final grid
                "comments": comments, # Comments from the final grid
                "predicted_price": predicted_price,
                "upper_bound": upper_bound,
                "actual_price": actual_price
            }

            calculated_price = round(predicted_price, 2) if predicted_price is not None else None
            adjusted_price = round(upper_bound, 2) if upper_bound is not None else None
            actual_price = round(actual_price, 2) if actual_price is not None else None

            status_message = "Calculation Completed"
            if deleted_count_current_request > 0:
                 status_message += f". {deleted_count_current_request} points deleted in this step."

        except Exception as e:
            print(f"Error during graph calculation: {e}")
            status_message = f"Error: An unexpected error occurred during calculation: {e}"
            calculated_price = adjusted_price = actual_price = "Error"
            chart_data = {} # Ensure empty chart data on error

        # Render template with results and the inputs that were used
        # Use discogs_data for the textarea display as it was submitted
        pasted_discogs_data_display = discogs_data

        return render_template("index.html",
                               pasted_discogs_data=pasted_discogs_data_display,
                               media=media_display,
                               sleeve=sleeve_display,
                               shop_var=shop_var_display,
                               start_date=start_date_display,
                               add_data=add_data_display,
                               calculated_price=calculated_price,
                               adjusted_price=adjusted_price,
                               actual_price=actual_price,
                               chart_data=chart_data,
                               status_message=status_message)


    # This block is for the initial GET request
    print("GET request. Loading saved shop_var for display.")
    shop_var_display = read_shop_var() # Load initial shop_var

    # Clear session data on initial GET to start fresh for a new user session
    # This ensures that pasting new data or refreshing the page starts a new calculation basis.
    if 'processed_grid' in session:
         del session['processed_grid']

    # Render template with default values
    return render_template("index.html",
                           pasted_discogs_data=pasted_discogs_data_display, # Default empty
                           media=media_display, # Default 6
                           sleeve=sleeve_display, # Default 6
                           shop_var=shop_var_display, # Loaded from file
                           start_date=start_date_display, # Default
                           add_data=add_data_display, # Default False
                           status_message="", # No status on initial load
                           calculated_price=None,
                           adjusted_price=None,
                           actual_price=None,
                           chart_data={})

if __name__ == "__main__":
    # In production, use a production-ready WSGI server like Gunicorn or uWSGI
    # app.run(debug=True, port=5002) # Set debug=False for production
    # Setting debug=False for deployment environment
    threading.Thread(target=open_browser_once).start() # Uncomment to auto-open browser on start
    app.run(debug=False, port=5002)