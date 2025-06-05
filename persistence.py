# /home/stigoftdump/PycharmProjects/PythonProject/vinylprice/persistence.py
import pickle
import sys
import os

unified_save_file = "vinylpricedata.pkl"
ml_save_file = "ml_data_pkl" # Consider renaming to ml_sales_data.pkl for clarity if you like

def read_application_data():
    """Reads the entire data dictionary from the unified pickle file."""
    try:
        with open(unified_save_file, 'rb') as f:
            data = pickle.load(f)
            return data
    except (FileNotFoundError, EOFError, pickle.UnpicklingError): # Handle empty or corrupted file
        return {} # Return an empty dict if file not found or unpickling error

def write_application_data(data):
    """Writes the entire data dictionary to the unified pickle file."""
    try:
        with open(unified_save_file, 'wb') as f:
            pickle.dump(data, f)
    except IOError as e:
        print(f"Error writing to {unified_save_file}: {e}", file=sys.stderr)

def read_save_value(datatype, default):
    """
    Reads the data type from the saved file

    It returns the default if it cant read the file

    """
    data = read_application_data()
    datareturn = data.get(datatype, default)

    return datareturn

def write_save_value(value, datatype):
    """
    Write the value to the datatype in the save file
    """
    data = read_application_data() # Read existing data
    data[datatype] = value
    write_application_data(data)

def read_ml_data():
    """
    Reads the accumulated ML sales data.

    Returns:
        list: A list of dictionaries, each representing a sale.
              Returns an empty list if the file doesn't exist or is empty/corrupted.
    """
    sales_list = []
    if os.path.exists(ml_save_file):
        try:
            with open(ml_save_file, 'rb') as f:
                sales_list = pickle.load(f)
                if not isinstance(sales_list, list): # Basic type check
                    print(f"Warning: ML data file '{ml_save_file}' did not contain a list. Starting with empty ML data.", file=sys.stderr)
                    sales_list = []
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Warning: Could not read or unpickle ML data file '{ml_save_file}': {e}. Starting with empty ML data.", file=sys.stderr)
            sales_list = []
        except Exception as e:
            print(f"Error reading ML data file '{ml_save_file}': {e}", file=sys.stderr)
            sales_list = []
    return sales_list

def write_ml_data(sales_list):
    """
    Writes the accumulated ML sales data to a file.

    Args:
        sales_list (list): The list of dictionaries containing sale data for ML.
    """
    try:
        with open(ml_save_file, 'wb') as f:
            pickle.dump(sales_list, f)
    except Exception as e:
        print(f"Error writing ML data file '{ml_save_file}': {e}", file=sys.stderr)