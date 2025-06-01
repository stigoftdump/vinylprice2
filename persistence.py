import pickle

unified_save_file = "vinylpricedata.pkl"

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
    Reads the data type from the saved file.
    It returns the default if it cant read the file.
    """
    data = read_application_data()
    datareturn = data.get(datatype, default)
    return datareturn

def write_save_value(value, datatype):
    """
    Write the value to the datatype in the save file.
    """
    data = read_application_data() # Read existing data
    data[datatype] = value
    write_application_data(data)