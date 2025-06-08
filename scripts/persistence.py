# /home/stigoftdump/PycharmProjects/PythonProject/vinylprice/persistence.py
import pickle
import sys
import os

# Get the directory of the current script (e.g., /path/to/your_project_root/scripts)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (one level up from scripts, e.g., /path/to/your_project_root)
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# Define paths as absolute paths from the project root
unified_save_file = os.path.join(_PROJECT_ROOT, "vinylpricedata.pkl")
ml_save_file = os.path.join(_PROJECT_ROOT, "ml_data.pkl")


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

def remember_last_run(processed_grid, discogs_id, artist, title, year, original_year):
    """
    Saves multiple pieces of data from the last run to the unified save file
    efficiently by performing a single read and a single write operation.
    """
    # Read the entire application data once
    data = read_application_data()

    # Update all the necessary keys in the dictionary
    data["processed_grid"] = processed_grid

    if discogs_id is not None:
        data["last_discogs_id"] = discogs_id
    if artist is not None:
        data["last_artist"] = artist
    if title is not None:
        data["last_title"] = title
    if year is not None:
        data["last_year"] = year
    if original_year is not None:
        data["last_original_year"] = original_year  # Corrected to use the original_year parameter

    # Write the updated dictionary back once
    write_application_data(data)

def recall_last_run():
    # gets the last run from
    data = read_application_data()

    recalled_data = {}

    recalled_data["processed_grid"] = data.get("processed_grid")
    recalled_data["discogs_id"] = data.get("last_discogs_id")
    recalled_data["artist"] = data.get("last_artist")
    recalled_data["title"] = data.get("last_title")
    recalled_data["year"] = data.get("last_year")
    recalled_data["original_year"] = data.get("last_original_year")

    return recalled_data

def machine_learning_save(processed_grid, discogs_release_id=None, api_data=None):
    """
    Saves or updates release information and its associated sales data to the ML data file.
    The ML data is structured as a dictionary keyed by discogs_release_id.
    Each release entry contains API metadata and a list of its sales history.
    Sales data is de-duplicated based on date, quality score, and native price.
    """
    all_releases_data = read_ml_data()

    if not isinstance(all_releases_data, dict):
        print("Warning: ML data file is not a dictionary. Initializing as empty. "
              "If you have old list-based data, it needs migration.", file=sys.stderr)
        all_releases_data = {}

    if not discogs_release_id:
        if processed_grid:  # Only warn if there was sales data that couldn't be associated
            print("Warning: Sales data provided but no discogs_release_id. "
                  "These sales will not be saved in the release-centric ML data structure.", file=sys.stderr)
        # If no ID and no sales, it's fine, just nothing to do.
        return

    release_key = str(discogs_release_id)
    new_sales_added_count = 0
    api_data_updated_flag = False

    release_entry = all_releases_data.get(release_key)

    if release_entry is None:
        release_entry = {}
        if api_data:
            release_entry.update(api_data)
            api_data_updated_flag = True  # API data added for new entry
        release_entry['sales_history'] = []
        all_releases_data[release_key] = release_entry
        print(f"Info: Creating new ML data entry for Release ID: {release_key}", file=sys.stderr)
    else:
        if api_data:
            # Check if API data has actually changed before updating and setting flag
            release_entry.update(api_data)  # Simpler: always update if api_data is present
            api_data_updated_flag = True
            print(f"Info: Updating/refreshing API data for existing Release ID: {release_key}", file=sys.stderr)

        if 'sales_history' not in release_entry or not isinstance(release_entry['sales_history'], list):
            release_entry['sales_history'] = []

    existing_sale_identifiers_for_release = set()
    for sale in release_entry['sales_history']:
        sale_identifier = (
            sale.get('date', ''),
            round(sale.get('quality', 0.0), 5),  # Round for consistent comparison
            str(sale.get('native_price', ''))  # Ensure native_price is string for consistency
        )
        existing_sale_identifiers_for_release.add(sale_identifier)

    current_batch_sale_identifiers = set()

    if processed_grid:
        for row in processed_grid:
            if len(row) >= 7:
                sale_date = row[0]
                native_price_from_row = str(row[4] or '')  # Ensure string, handle None
                quality_score = row[5]
                inflation_adjusted_price = row[3]
                sale_specific_comment = row[6]

                current_sale_de_dup_key = (
                    sale_date or '',
                    round(quality_score, 5),
                    native_price_from_row
                )

                if current_sale_de_dup_key in existing_sale_identifiers_for_release or \
                        current_sale_de_dup_key in current_batch_sale_identifiers:
                    continue

                current_batch_sale_identifiers.add(current_sale_de_dup_key)
                sale_dict = {
                    'date': sale_date,
                    'quality': quality_score,
                    'price': inflation_adjusted_price,
                    'native_price': native_price_from_row,
                    'sale_comment': sale_specific_comment
                }
                release_entry['sales_history'].append(sale_dict)
                new_sales_added_count += 1
            else:
                print(f"Warning: Skipping row with unexpected structure for ML data: {row}", file=sys.stderr)

    # Determine if a write is needed: if new sales were added OR if API data was updated/added.
    if new_sales_added_count > 0 or api_data_updated_flag:
        write_ml_data(all_releases_data)
        artist_name_for_log = release_entry.get('api_artist', 'N/A')
        album_title_for_log = release_entry.get('api_title', 'N/A')
        log_message_parts = []
        if api_data_updated_flag:
            log_message_parts.append("API data processed/updated")
        if new_sales_added_count > 0:
            log_message_parts.append(f"added {new_sales_added_count} new sales")

        print(f"Info: For Release ID '{release_key}' ({artist_name_for_log} - {album_title_for_log}): "
              f"{' and '.join(log_message_parts)}. ML data file updated.", file=sys.stderr)
    else:
        artist_name_for_log = release_entry.get('api_artist', 'N/A') if release_entry else 'N/A'
        album_title_for_log = release_entry.get('api_title', 'N/A') if release_entry else 'N/A'
        print(f"Info: No new unique sales or API data updates for Release ID '{release_key}' "
              f"({artist_name_for_log} - {album_title_for_log}). ML data file not modified for this entry.",
              file=sys.stderr)





