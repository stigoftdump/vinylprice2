from flask import Flask, render_template, request
import webbrowser
import threading
import time
from vin import calculate_vin_data
from persistence import read_save_value, write_save_value
import os

app = Flask(__name__)

def open_browser_once():
    """Wait briefly and open the browser."""
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:5002")

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

    # Changes depending on where it's POST or GET
    if request.method == "POST":
        # Read form values to repopulate when reloading.
        media = int(request.form.get("media", 6))
        sleeve = int(request.form.get("sleeve", 6))
        shop_var = float(request.form.get("shop_var", 0.8))
        start_date = request.form.get("start_Date", "2020-01-01")
        add_data = request.form.get("add_data", "off")
        add_data_flag = True if add_data == "on" else False
        points_to_delete_json = request.form.get('selected_points_to_delete', '[]')
        discogs_data = request.form.get('pasted_discogs_data', '') # Always get from form

        # Put them into variables for displaying at POST render
        media_display = media
        sleeve_display = sleeve
        shop_var_display = shop_var
        start_date_display = start_date
        add_data_display = False # set to false every time it runs so the user has to check it each time

        # Calculate quality from media and sleeve
        quality = media - ((media - sleeve) / 3)

        # Save the shop_var value for next session
        write_save_value(shop_var, "shop_var")

        # Calls the vin data function, it all gets returned as json
        status_message = "Calculating"

        output_data = calculate_vin_data(
            quality,
            shop_var,
            start_date,
            str(add_data_flag), # Pass boolean as string as it was expected by the original logic
            discogs_data,
            points_to_delete_json
        )

        # assigns the data to variables
        calculated_price = output_data.get("calculated_price")
        adjusted_price = output_data.get("upper_bound")
        actual_price = output_data.get("actual_price")
        status_message = output_data.get("status_message", status_message)
        info_message = output_data.get("info_message")
        error_message = output_data.get("error_message")
        chart_data = output_data.get("chart_data", {})

        # Render template with results and the inputs that were used
        # Populate form fields with the submitted values for continuity
        return render_template("index.html",
                               pasted_discogs_data='',
                               media=media_display,
                               sleeve=sleeve_display,
                               shop_var=shop_var_display,
                               start_date=start_date_display,
                               add_data=add_data_display,
                               calculated_price=calculated_price,
                               adjusted_price=adjusted_price,
                               actual_price=actual_price,
                               chart_data=chart_data,
                               status_message=status_message,
                               info_message=info_message,
                               error_message=error_message,
                               is_initial_load = False)  # Indicate not initial load

    else:
        # This block is for the initial GET request

        # Variables loaded from
        media_display = read_save_value("media_quality", 6)
        sleeve_display = read_save_value("sleeve_quality", 6)
        shop_var_display = read_save_value("shop_var", 0.8)
        start_date_display = read_save_value("start_date", "2020-01-01")
        add_data_display = False

        # Renders the initial template
        return render_template("index.html",
                               pasted_discogs_data='',
                               media=media_display,
                               sleeve=sleeve_display,
                               shop_var=shop_var_display,
                               start_date=start_date_display,
                               add_data=add_data_display,
                               status_message="",
                               info_message=None,
                               error_message=None,
                               calculated_price=None,
                               adjusted_price=None,
                               actual_price=None,
                               chart_data={},
                               is_initial_load=True) # indicates initial load for the splash screen

if __name__ == "__main__":
    #from werkzeug.serving import is_running_from_reloader

    #if not is_running_from_reloader():
    threading.Thread(target=open_browser_once).start()
    app.run(debug=False, port=5002)