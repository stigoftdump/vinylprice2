from flask import Flask, render_template, request, send_from_directory
import subprocess
import os
import json
import webbrowser
import threading
import time

app = Flask(__name__)

# File to store saved value
SAVE_FILE = "save_data.json"

def open_browser_once():
    """Wait briefly and open the browser."""
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:5002")

def read_save_value():
    try:
        with open(SAVE_FILE, 'r') as f:
            data = json.load(f)
            return data.get("shop_var", 0.8)  # Default to 0.8 if not set
    except (FileNotFoundError, json.JSONDecodeError):
        return 0.8  # Default to 0.8

def write_save_value(value):
    with open(SAVE_FILE, 'w') as f:
        json.dump({"shop_var": value}, f)

@app.route("/", methods=["GET", "POST"])
def index():
    calculated_price = None
    adjusted_price = None
    actual_price = None
    max_price = None

    # Load shop_var from saved file or use default 0.8
    shop_var = read_save_value()

    if request.method == "POST":

        # gets the discogs data
        discogs_data = request.form.get('pasted_discogs_data')

        # Get the quality input from the form
        media = int(request.form["media"])
        sleeve = int(request.form["sleeve"])
        quality = media - ((media - sleeve) / 3)

        #shop_var = request.form["shop_var"]  # Capture the shop_var value from the form
        write_save_value(shop_var)  # Save the new value
        shop_var = request.form["shop_var"]  # Capture the shop_var value from the form

        start_date = request.form["start_Date"] # Capture the start_Date from the form
        add_data = request.form.get("add_data", "off")  # Use .get() to avoid KeyError
        add_data_flag = True if add_data == "on" else False  # Convert to a boolean

        # gets the max price
        max_price = request.form.get("max_price","").strip()

        # Use the full path to the Python interpreter in your virtual environment
        project_path = '/home/stigoftdump/PycharmProjects/PythonProject/vinylprice/'
        python_path = project_path + '.venv/bin/python'  # Correct path
        code_path = project_path + 'vin.py'

        # Run the Python script with the input
        result = subprocess.run([python_path, code_path, str(quality), str(shop_var), start_date, str(add_data_flag), str(max_price), discogs_data], capture_output=True, text=True)

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
            calculated_price = adjusted_price = actual_price = "Error"
        else:
            try:
                # Debugging split and float conversion
                calculated_price, adjusted_price, actual_price = map(float, output.split(","))
            except ValueError:
                # Handle cases where the script output is invalid
                print("Error parsing the output.")
                calculated_price = adjusted_price = actual_price = "Error"

            # Return the prices and image path to be displayed
        return render_template("index.html",
                               calculated_price=calculated_price,
                               adjusted_price=adjusted_price,
                               actual_price=actual_price,
                               media=media,
                               sleeve=sleeve,
                               chart_url="static/chart.png",
                               shop_var=shop_var,
                               max_price = max_price)

    # this is loaded on first run (GET) rather than POST
    return render_template("index.html",
                           calculated_price=calculated_price,
                           adjusted_price=adjusted_price,
                           actual_price=actual_price,
                           media=6,
                           sleeve=6,
                           chart_url=None,
                           shop_var = shop_var)

# To serve the image properly from the static folder
@app.route("/static/images/<filename>")
def send_image(filename):
    return send_from_directory(os.path.join(app.root_path, "static/images"), filename)


if __name__ == "__main__":
    from werkzeug.serving import is_running_from_reloader

    if not is_running_from_reloader():
        threading.Thread(target=open_browser_once).start()
    app.run(debug=True, port=5002)
