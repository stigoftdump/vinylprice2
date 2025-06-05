import pickle
import sys
import os

unified_save_file = "vinylpricedata.pkl"
ml_save_file = "ml_data_pkl"

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
        print(f"Error writing to {unified_save_file}: {e}")

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
    Reads the accumulated ML sales data and the last used record ID.

    Returns:
        tuple: A tuple containing:
            - sales_list (list): A list of dictionaries, each representing a sale
                                 with 'record_id', 'date', 'quality', 'price'.
            - last_record_id (int): The last record ID that was assigned.
                                    Returns ([], 0) if the file doesn't exist or is empty/corrupted.
    """
    data = {'sales': [], 'last_record_id': 0}
    if os.path.exists(ml_save_file):
        try:
            with open(ml_save_file, 'rb') as f:
                loaded_data = pickle.load(f)
                # Ensure loaded data has the expected structure, provide defaults if not
                data['sales'] = loaded_data.get('sales', [])
                data['last_record_id'] = loaded_data.get('last_record_id', 0)
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Warning: Could not read or unpickle ML data file '{ml_save_file}': {e}. Starting with empty ML data.", file=sys.stderr)
            # data remains {'sales': [], 'last_record_id': 0}
        except Exception as e:
            print(f"Error reading ML data file '{ml_save_file}': {e}", file=sys.stderr)
            # data remains {'sales': [], 'last_record_id': 0}
    return data['sales'], data['last_record_id']

def write_ml_data(sales_list, last_record_id):
    """
    Writes the accumulated ML sales data and the last used record ID to a file.

    Args:
        sales_list (list): The list of dictionaries containing sale data for ML.
        last_record_id (int): The last record ID that was assigned.
    """
    data = {'sales': sales_list, 'last_record_id': last_record_id}
    try:
        with open(ml_save_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error writing ML data file '{ml_save_file}': {e}", file=sys.stderr)
