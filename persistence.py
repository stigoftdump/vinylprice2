# /home/stigoftdump/PycharmProjects/PythonProject/vinylprice/persistence.py
import pickle
import sys
import os

unified_save_file = "vinylpricedata.pkl"
ml_save_file = "ml_data.pkl"  # Changed from ml_data_pkl to ml_data.pkl for consistency


def read_application_data():
    """Reads the entire data dictionary from the unified pickle file."""
    try:
        with open(unified_save_file, 'rb') as f:
            data = pickle.load(f)
            # Ensure it's a dictionary, return empty if not (e.g. corrupted file)
            if not isinstance(data, dict):
                print(
                    f"Warning: Unified save file '{unified_save_file}' did not contain a dictionary. Returning empty data.",
                    file=sys.stderr)
                return {}
            return data
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):  # Handle empty or corrupted file
        return {}  # Return an empty dict if file not found or unpickling error
    except Exception as e:
        print(f"Error reading unified save file '{unified_save_file}': {e}. Returning empty data.", file=sys.stderr)
        return {}


def write_application_data(data):
    """Writes the entire data dictionary to the unified pickle file."""
    if not isinstance(data, dict):
        print(f"Error: Attempted to write non-dictionary data to {unified_save_file}. Aborting write.", file=sys.stderr)
        return
    try:
        with open(unified_save_file, 'wb') as f:
            pickle.dump(data, f)
    except IOError as e:
        print(f"Error writing to {unified_save_file}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred while writing to {unified_save_file}: {e}", file=sys.stderr)


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
    data = read_application_data()  # Read existing data
    data[datatype] = value
    write_application_data(data)


def read_ml_data():
    """
    Reads the accumulated ML data, which is expected to be a dictionary
    keyed by discogs_release_id.

    Returns:
        dict: A dictionary where keys are discogs_release_ids and values
              are dictionaries containing release metadata and sales history.
              Returns an empty dictionary {} if the file doesn't exist,
              is empty, corrupted, or not in the expected dictionary format.
    """
    if os.path.exists(ml_save_file):
        try:
            with open(ml_save_file, 'rb') as f:
                # Handle empty file case before attempting to load
                if os.fstat(f.fileno()).st_size == 0:
                    return {}  # Return empty dict for an empty file

                ml_data_content = pickle.load(f)
                if not isinstance(ml_data_content, dict):
                    print(f"Warning: ML data file '{ml_save_file}' did not contain a dictionary. "
                          f"This might be old list-based data. Starting with empty ML data. "
                          f"Manual migration might be needed for old data.", file=sys.stderr)
                    return {}
                return ml_data_content
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Warning: Could not read or unpickle ML data file '{ml_save_file}': {e}. "
                  f"Starting with empty ML data.", file=sys.stderr)
            return {}
        except Exception as e:  # Catch other potential errors during file read or unpickling
            print(f"Error reading ML data file '{ml_save_file}': {e}. "
                  f"Starting with empty ML data.", file=sys.stderr)
            return {}
    return {}  # File does not exist, return empty dict


def write_ml_data(ml_releases_data):
    """
    Writes the ML data (dictionary of releases) to a file.

    Args:
        ml_releases_data (dict): The dictionary containing ML data,
                                 keyed by discogs_release_id.
    """
    if not isinstance(ml_releases_data, dict):
        print(f"Error: Attempted to write non-dictionary data to ML data file '{ml_save_file}'. Aborting write.",
              file=sys.stderr)
        return
    try:
        with open(ml_save_file, 'wb') as f:
            pickle.dump(ml_releases_data, f)
    except Exception as e:
        print(f"Error writing ML data file '{ml_save_file}': {e}", file=sys.stderr)
