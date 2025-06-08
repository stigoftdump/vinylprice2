from flask import Flask, render_template, request
import webbrowser
import threading
import time
from scripts.vin import calculate_vin_data
from scripts.persistence import read_save_value, write_save_value
import os
from urllib.parse import urlparse  # For parsing the URL
import re  # For regex matching of the ID

app = Flask(__name__)

# Turn on debug mode for making it easier when writing
debug_mode = True

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


def parse_discogs_url_for_id(url_string):
    """
    Parses a Discogs URL string to extract the release ID.
    Handles URLs like:
    - https://www.discogs.com/sell/history/4242486
    - https://www.discogs.com/release/4242486-Artist-AlbumTitle
    - https://www.discogs.com/Artist-AlbumTitle/release/4242486
    - Just the ID itself: 4242486

    Args:
        url_string (str): The Discogs URL or just an ID.

    Returns:
        str: The extracted release ID as a string, or None if not found or invalid.
    """
    if not url_string:
        return None

    # Check if the string is just a number (the ID itself)
    if url_string.isdigit():
        return url_string

    try:
        # Regex to find a sequence of digits at the end of a path segment,
        # or between /release/ and the next segment.
        # This covers /history/ID, /release/ID-..., .../release/ID
        match = re.search(r'(?:/release/|/history/)(\d+)(?:[/?#]|$|-)', url_string)
        if match:
            return match.group(1)

        # Fallback for URLs like /Artist-Album/release/ID
        match_alternative = re.search(r'/release/(\d+)$', url_string)
        if match_alternative:
            return match_alternative.group(1)

        # If the above don't match, try a more general approach for IDs at the end of any path
        parsed_url = urlparse(url_string)
        path_segments = parsed_url.path.strip('/').split('/')
        if path_segments:
            last_segment = path_segments[-1]
            # Check if the last segment is purely numeric (an ID)
            if last_segment.isdigit():
                return last_segment
            # Check if the last segment starts with digits followed by a hyphen (e.g., 12345-Artist-Title)
            id_match_in_segment = re.match(r'^(\d+)-', last_segment)
            if id_match_in_segment:
                return id_match_in_segment.group(1)
    except Exception:
        # If any parsing error occurs, return None
        return None
    return None


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
    # API data placeholders
    api_artist_display = None
    api_title_display = None
    api_year_display = None
    api_original_year_display = None

    # Changes depending on where it's POST or GET
    if request.method == "POST":
        # Read form values
        media = int(request.form.get("media", 6))
        sleeve = int(request.form.get("sleeve", 6))
        shop_var = float(request.form.get("shop_var", 0.8))
        start_date = request.form.get("start_Date", "2020-01-01")
        add_data = request.form.get("add_data", "off")
        add_data_flag = True if add_data == "on" else False
        points_to_delete_json = request.form.get('selected_points_to_delete', '[]')
        discogs_data = request.form.get('pasted_discogs_data', '')
        discogs_url_from_form = request.form.get('discogs_url', '')
        discogs_release_id = parse_discogs_url_for_id(discogs_url_from_form)

        # Variables for displaying at POST render (persisted values)
        media_display = media
        sleeve_display = sleeve
        shop_var_display = shop_var
        start_date_display = start_date
        add_data_display = False  # set to false every time it runs

        quality = media - ((media - sleeve) / 3)
        write_save_value(shop_var, "shop_var")
        status_message = "Calculating"

        output_data = calculate_vin_data(
            quality,
            shop_var,
            start_date,
            str(add_data_flag),
            discogs_data,
            points_to_delete_json,
            discogs_release_id
        )

        calculated_price = output_data.get("calculated_price")
        adjusted_price = output_data.get("upper_bound")
        actual_price = output_data.get("actual_price")
        status_message = output_data.get("status_message", status_message)
        info_message = output_data.get("info_message")
        error_message = output_data.get("error_message")
        chart_data = output_data.get("chart_data", {})
        api_artist_display = output_data.get("api_artist")
        api_title_display = output_data.get("api_title")
        api_year_display = output_data.get("api_year")
        api_original_year_display = output_data.get("api_original_year")


        return render_template("index.html",
                               pasted_discogs_data='',
                               media=media_display,
                               sleeve=sleeve_display,
                               shop_var=shop_var_display,
                               start_date=start_date_display,
                               add_data=add_data_display,
                               discogs_url='',  # Field is cleared
                               calculated_price=calculated_price,
                               adjusted_price=adjusted_price,
                               actual_price=actual_price,
                               chart_data=chart_data,
                               status_message=status_message,
                               info_message=info_message,
                               error_message=error_message,
                               api_artist=api_artist_display,
                               api_title=api_title_display,
                               api_year=api_year_display,
                               api_original_year=api_original_year_display,
                               is_initial_load=False)

    else:  # GET request
        media_display = read_save_value("media_quality", 6)
        sleeve_display = read_save_value("sleeve_quality", 6)
        shop_var_display = read_save_value("shop_var", 0.8)
        start_date_display = read_save_value("start_date", "2020-01-01")
        add_data_display = False

        return render_template("index.html",
                               pasted_discogs_data='',
                               media=media_display,
                               sleeve=sleeve_display,
                               shop_var=shop_var_display,
                               start_date=start_date_display,
                               add_data=add_data_display,
                               discogs_url='',  # Field is empty on initial load
                               status_message="",
                               info_message=None,
                               error_message=None,
                               calculated_price=None,
                               adjusted_price=None,
                               actual_price=None,
                               chart_data={},
                               api_artist=None,
                               api_title=None,
                               api_year=None,
                               api_original_year=None,
                               is_initial_load= not debug_mode) # only load the splashscreen if not in debug mode

if __name__ == "__main__":
    threading.Thread(target=open_browser_once).start()
    app.run(debug=debug_mode, port=5002)
