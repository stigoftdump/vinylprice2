from flask import Flask, render_template, request, send_from_directory
import json
import webbrowser
import threading
import time
from datetime import datetime # Import datetime
from vin import calculate_vin_data
import os


app = Flask(__name__)

# File to store saved shop_var
SAVE_FILE = "save_data.json"

def open_browser_once():
    """Wait briefly and open the browser."""
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:5002")

# --- Keep existing shop_var functions ---
def read_save_value():
    """
    Reads the 'shop_var' value from the save file.

    Returns:
        float: The saved 'shop_var' value, or 0.8 as a default if the file
               doesn't exist, is invalid, or 'shop_var' is not set.
    """
    try:
        with open(SAVE_FILE, 'r') as f:
            data = json.load(f)
            return data.get("shop_var", 0.8)  # Default to 0.8 if not set
    except (FileNotFoundError, json.JSONDecodeError):
        return 0.8  # Default to 0.8

def write_save_value(value):
    """
    Writes the given 'shop_var' value to the save file.

    Args:
        value (float): The 'shop_var' value to save.
    """
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
    os._exit(0)

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Handles GET and POST requests for the main page.

    GET: Renders the main page with default or saved values.
    POST: Processes the form submission, calculates the price using 'vin.py',
          and renders the page with the results and updated form values.

    Returns:
        str: Rendered HTML template ('index.html') with context variables.
    """
    calculated_price = None
    adjusted_price = None
    actual_price = None
    status_message = ""
    chart_data = {}  # Initialize chart_data

    # Variables to hold form values for rendering (will be populated below)
    pasted_discogs_data_display = ""
    media_display = 6
    sleeve_display = 6
    shop_var_display = read_save_value() # Load initial shop_var for GET
    start_date_display = "2020-01-01"
    add_data_display = False

    if request.method == "POST":
        # Determine which button was clicked
        action = request.form.get('action')

        # --- Read form values regardless of action, except discogs_data for rerun ---
        media = int(request.form.get("media", 6)) # Provide default
        sleeve = int(request.form.get("sleeve", 6)) # Provide default
        shop_var = float(request.form.get("shop_var", 0.8)) # Provide default
        start_date = request.form.get("start_Date", "2020-01-01") # Provide default
        add_data = request.form.get("add_data", "off")
        add_data_flag = True if add_data == "on" else False
        points_to_delete_json = request.form.get('selected_points_to_delete', '[]')

        # use the saved data is possible, otherwise the pasted.
        if request.form.get('saved_discogs_data'):
            discogs_data = request.form.get('saved_discogs_data', '')
        else:
            discogs_data = request.form.get('pasted_discogs_data', '')

        # Update display variables with submitted values for rendering
        media_display = media
        sleeve_display = sleeve
        shop_var_display = shop_var
        start_date_display = start_date
        add_data_display = add_data_flag

        # Calculate quality from media and sleeve
        quality = media - ((media - sleeve) / 3)

        # Save the shop_var value for next session (using original save mechanism)
        write_save_value(shop_var)

        # Validate date format briefly
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
        except ValueError:
             status_message = "Invalid start date format. Please use ISO YYYY-MM-DD."
             # Render template with current form values and error message
             return render_template("index.html",
                       pasted_discogs_data="", # Keep the discogs data in the textarea
                       media=media_display,
                       sleeve=sleeve_display,
                       shop_var=shop_var_display,
                       start_date=start_date_display,
                       add_data=add_data_display, # Pass boolean
                       status_message=status_message,
                       calculated_price=None, adjusted_price=None, actual_price=None, chart_data={})


        # --- Prepare vin.py arguments based on action ---
        print("Calculate button clicked. Using Discogs data from form.")
        process_discogs_data = discogs_data # Use form data for calculate
        pasted_discogs_data_display = discogs_data # Keep it in the textarea after calc
        status_message = "Calculating"

        # --- Call the function directly ---
        try:
            output_data = calculate_vin_data(
                quality,
                shop_var,
                start_date,
                str(add_data_flag), # Pass boolean as string as it was expected by the original logic
                process_discogs_data,
                points_to_delete_json
            )
            calculated_price = output_data.get("calculated_price")
            adjusted_price = output_data.get("upper_bound")
            actual_price = output_data.get("actual_price")
            status_message = output_data.get("status_message", status_message)
            chart_data = output_data.get("chart_data", {})

        except Exception as e:
            print(f"Error calling vin.py function: {e}")
            status_message = f"Error: An unexpected error occurred during calculation: {e}"
            calculated_price = adjusted_price = actual_price = "Error"
            chart_data = {} # Ensure empty chart data on error
        # --- End of direct function call ---


        # Render template with results and the inputs that were used
        # Populate form fields with the submitted values for continuity
        return render_template("index.html",
                               pasted_discogs_data=pasted_discogs_data_display, # Keep the discogs data in the textarea
                               media=media_display,
                               sleeve=sleeve_display,
                               shop_var=shop_var_display,
                               start_date=start_date_display,
                               add_data=add_data_display, # Pass boolean
                               calculated_price=calculated_price,
                               adjusted_price=adjusted_price,
                               actual_price=actual_price,
                               chart_data=chart_data,
                               status_message=status_message)

    # This block is for the initial GET request
    # Load shop_var for display, other defaults are hardcoded or handled by Jinja
    print("GET request. Loading saved shop_var for display.")
    shop_var_display = read_save_value() # Load initial shop_var

    # Render template with default values (or potentially last values if they were POSTed)
    # Jinja will use the default values passed here on the first GET
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
    #from werkzeug.serving import is_running_from_reloader

    #if not is_running_from_reloader():
    threading.Thread(target=open_browser_once).start()
    app.run(debug=False, port=5002)