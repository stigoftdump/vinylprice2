from flask import Flask, render_template, request, send_from_directory
import subprocess
import os
import json
import webbrowser
import threading
import time
from datetime import datetime # Import datetime

app = Flask(__name__)

# File to store saved shop_var
SAVE_FILE = "save_data.json"

def open_browser_once():
    """Wait briefly and open the browser."""
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:5002")

# --- Keep existing shop_var functions ---
def read_save_value():
    try:
        with open(SAVE_FILE, 'r') as f:
            data = json.load(f)
            return data.get("shop_var", 0.8)  # Default to 0.8 if not set
    except (FileNotFoundError, json.JSONDecodeError):
        return 0.8  # Default to 0.8

def write_save_value(value):
    try:
        with open(SAVE_FILE, 'w') as f:
            json.dump({"shop_var": value}, f)
    except IOError as e:
        print(f"Error writing to {SAVE_FILE}: {e}")


@app.route("/", methods=["GET", "POST"])
def index():
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
    max_price_display = ""


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
        max_price = request.form.get("max_price", "").strip()
        discogs_data = request.form.get('pasted_discogs_data', '') # Always get from form

        # Update display variables with submitted values for rendering
        media_display = media
        sleeve_display = sleeve
        shop_var_display = shop_var
        start_date_display = start_date
        add_data_display = add_data_flag
        max_price_display = max_price
        # pasted_discogs_data_display is handled below based on action


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
                       pasted_discogs_data=discogs_data, # Keep the discogs data in the textarea
                       media=media_display,
                       sleeve=sleeve_display,
                       shop_var=shop_var_display,
                       start_date=start_date_display,
                       add_data=add_data_display, # Pass boolean
                       max_price=max_price_display,
                       status_message=status_message,
                       calculated_price=None, adjusted_price=None, actual_price=None, chart_data={})


        # --- Prepare vin.py arguments based on action ---
        process_discogs_data = "" # Default to empty for rerun
        if action == "calculate":
            print("Calculate button clicked. Using Discogs data from form.")
            process_discogs_data = discogs_data # Use form data for calculate
            pasted_discogs_data_display = discogs_data # Keep it in the textarea after calc
            status_message = "Calculating"
        elif action == "rerun":
            print("Rerun button clicked. Using saved processed data.")
            # process_discogs_data remains ""
            # pasted_discogs_data_display keeps its value (last used data)
            status_message = "Rerunning with saved data."
        else:
            # Should not happen with button names, but good fallback
            status_message = "Unknown action."
            print(f"Unknown action: {action}")
            # Render template with current form values and error message
            return render_template("index.html",
                       pasted_discogs_data=discogs_data, # Keep the discogs data in the textarea
                       media=media_display,
                       sleeve=sleeve_display,
                       shop_var=shop_var_display,
                       start_date=start_date_display,
                       add_data=add_data_display, # Pass boolean
                       max_price=max_price_display,
                       status_message=status_message,
                       calculated_price=None, adjusted_price=None, actual_price=None, chart_data={})


        # Get the directory of the current script (this file)
        script_directory = os.path.dirname(os.path.abspath(__file__))

        # Construct the relative paths
        project_path = script_directory
        # Assuming .venv path is correct
        python_path = os.path.join(script_directory, '.venv', 'bin', 'python')
        code_path = os.path.join(script_directory, 'vin.py')

        # Construct vin.py arguments as strings
        vin_args = [
            python_path,
            code_path,
            str(quality),
            str(shop_var),
            start_date,
            str(add_data_flag), # Pass boolean as string
            str(max_price),
            process_discogs_data # This will be empty string for rerun
        ]

        print(f"Running vin.py with args: {vin_args}") # Debug print

        # Run the Python script with the input
        result = subprocess.run(vin_args, capture_output=True, text=True)

        # Capture and debug the output
        output = result.stdout.strip()
        error_output = result.stderr.strip()

        # Print raw output and error output for debugging
        print(f"Raw output from script: '{output}'")
        print(f"Error output from script: '{error_output}'")
        print(f"Return code: {result.returncode}")

        if result.returncode != 0:
            # Handle failure
            print(f"Script failed with error code {result.returncode}")
            status_message = f"Error: Script failed with exit code {result.returncode}. Check server logs."
            calculated_price = adjusted_price = actual_price = "Error"
            chart_data = {} # Ensure empty chart data on error
        else:
            try:
                # Load the JSON output
                output_data = json.loads(output)
                calculated_price = output_data.get("calculated_price")
                adjusted_price = output_data.get("upper_bound")
                actual_price = output_data.get("actual_price")
                # Prefer status message from script if available, otherwise use ours
                status_message = output_data.get("status_message", status_message) # Keep the rerun status message if no script error
                chart_data = output_data.get("chart_data", {})  # Extract chart data, default to empty dict

                # If script returned an error status message but didn't crash
                if output_data.get("status_message") and output_data.get("status_message") != "Completed":
                     status_message = output_data.get("status_message")
                     # Optionally clear results if the script indicated an issue
                     # calculated_price = adjusted_price = actual_price = None # Or "N/A"
                     # chart_data = {}

            except json.JSONDecodeError:
                # Handle cases where the script output is invalid JSON
                print(f"Error parsing JSON output from script: {output}")
                status_message = "Error: Could not parse script output."
                calculated_price = adjusted_price = actual_price = "Error"
                chart_data = {} # Ensure empty chart data on error
            except Exception as e:
                 # Catch any other potential errors during processing output
                 print(f"An unexpected error occurred processing script output: {e}")
                 status_message = f"Error: An unexpected error occurred: {e}"
                 calculated_price = adjusted_price = actual_price = "Error"
                 chart_data = {} # Ensure empty chart data on error


        # Render template with results and the inputs that were used
        # Populate form fields with the submitted values for continuity
        return render_template("index.html",
                               pasted_discogs_data=pasted_discogs_data_display, # Keep the discogs data in the textarea
                               media=media_display,
                               sleeve=sleeve_display,
                               shop_var=shop_var_display,
                               start_date=start_date_display,
                               add_data=add_data_display, # Pass boolean
                               max_price=max_price_display,
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
                           max_price=max_price_display, # Default empty
                           status_message="", # No status on initial load
                           calculated_price=None,
                           adjusted_price=None,
                           actual_price=None,
                           chart_data={})

# To serve the image properly from the static folder
@app.route("/static/images/<filename>")
def send_image(filename):
    return send_from_directory(os.path.join(app.root_path, "static/images"), filename)


if __name__ == "__main__":
    from werkzeug.serving import is_running_from_reloader

    if not is_running_from_reloader():
        threading.Thread(target=open_browser_once).start()
    app.run(debug=True, port=5002)