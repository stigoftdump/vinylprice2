from flask import Flask, render_template, request, send_from_directory
import subprocess
import os

app = Flask(__name__)

# File to store saved value
SAVE_FILE = "save_data.json"

def read_save_value():
    try:
        with open(SAVE_FILE, 'r') as f:
            data = json.load(f)
            return data.get("shop_var", 0.7)  # Default to 0.7 if not set
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
    if request.method == "POST":
        # Get the quality input from the form
        media = int(request.form["media"])
        sleeve = int(request.form["sleeve"])
        quality = media - ((media - sleeve) / 3)

        shop_var = request.form["shop_var"]  # Capture the shop_var value from the form

        # Use the full path to the Python interpreter in your virtual environment
        project_path = '/home/stigoftdump/PycharmProjects/PythonProject/vinylprice/'
        python_path = project_path + '.venv/bin/python'  # Correct path
        code_path = project_path + 'vin.py'

        # Run the Python script with the input
        #result = subprocess.run([python_path, code_path, str(quality), shop_var], capture_output=True, text=True)
        result = subprocess.run([python_path, code_path, str(quality), str(shop_var)], capture_output=True, text=True)

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
                               chart_url="static/images/chart.png")

    return render_template("index.html",
                           calculated_price=calculated_price,
                           adjusted_price=adjusted_price,
                           actual_price=actual_price,
                           media=6,
                           sleeve=6,
                           chart_url=None)


# To serve the image properly from the static folder
@app.route("/static/images/<filename>")
def send_image(filename):
    return send_from_directory(os.path.join(app.root_path, "static/images"), filename)


if __name__ == "__main__":
    app.run(debug=True)
