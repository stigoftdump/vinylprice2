# ml_import.py
import pickle
import os
import sys

# --- Configuration ---
EXISTING_DATA_FILE = "ml_data_pkl"
NEW_DATA_FILE = "ml_data_pkl_new"
BACKUP_DATA_FILE = "ml_data_pkl.bak"  # Optional: for backing up the original file

# Define the fields that constitute a unique record for de-duplication
# Adjust these if your definition of a unique sale entry differs
UNIQUE_KEY_FIELDS = ['artist', 'album', 'label', 'extra_comments', 'date', 'quality', 'price']


def load_pickle_data(filename):
    """Loads data from a pickle file."""
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                if not isinstance(data, list):
                    print(f"Warning: Data in {filename} is not a list. Initializing as empty list.", file=sys.stderr)
                    return []
                return data
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Error: Could not unpickle data from {filename}. It might be corrupted or empty. {e}",
                  file=sys.stderr)
            return []
        except Exception as e:
            print(f"An unexpected error occurred while loading {filename}: {e}", file=sys.stderr)
            return []
    return []


def save_pickle_data(data, filename):
    """Saves data to a pickle file."""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Successfully saved combined data to {filename}")
    except Exception as e:
        print(f"Error: Could not save data to {filename}. {e}", file=sys.stderr)


def create_record_key(record_dict):
    """Creates a tuple key from the record for de-duplication."""
    key_parts = []
    for field in UNIQUE_KEY_FIELDS:
        # Use a placeholder for missing fields to ensure consistent key structure
        key_parts.append(record_dict.get(field, None))
    return tuple(key_parts)


def main():
    print(f"Starting import process: appending '{NEW_DATA_FILE}' to '{EXISTING_DATA_FILE}'...")

    # 1. Load existing data
    existing_data = load_pickle_data(EXISTING_DATA_FILE)
    print(f"Loaded {len(existing_data)} records from '{EXISTING_DATA_FILE}'.")

    # 2. Load new data
    new_data = load_pickle_data(NEW_DATA_FILE)
    if not new_data:
        print(f"No data found in '{NEW_DATA_FILE}' or file does not exist. Nothing to append.")
        return
    print(f"Loaded {len(new_data)} records from '{NEW_DATA_FILE}'.")

    # 3. Create a set of keys from existing data for efficient duplicate checking
    existing_keys = set()
    for record in existing_data:
        if isinstance(record, dict):
            existing_keys.add(create_record_key(record))
        else:
            print(f"Warning: Found a non-dictionary item in existing data: {record}. Skipping for key generation.",
                  file=sys.stderr)

    # 4. Append new data, checking for duplicates
    appended_count = 0
    duplicate_count = 0
    malformed_new_data_count = 0

    combined_data = list(existing_data)  # Start with a copy of existing data

    for new_record in new_data:
        if not isinstance(new_record, dict):
            print(f"Warning: Found a non-dictionary item in new data: {new_record}. Skipping.", file=sys.stderr)
            malformed_new_data_count += 1
            continue

        new_key = create_record_key(new_record)
        if new_key not in existing_keys:
            combined_data.append(new_record)
            existing_keys.add(new_key)  # Add new key to prevent duplicates within new_data itself if any
            appended_count += 1
        else:
            duplicate_count += 1

    print(f"\n--- Import Summary ---")
    print(f"Records initially in '{EXISTING_DATA_FILE}': {len(existing_data)}")
    print(f"Records found in '{NEW_DATA_FILE}': {len(new_data)}")
    if malformed_new_data_count > 0:
        print(f"Malformed (non-dictionary) records skipped from '{NEW_DATA_FILE}': {malformed_new_data_count}")
    print(f"New unique records appended: {appended_count}")
    print(f"Duplicate records skipped: {duplicate_count}")
    print(f"Total records in combined dataset: {len(combined_data)}")

    # 5. Optional: Backup existing file before overwriting
    if os.path.exists(EXISTING_DATA_FILE) and appended_count > 0:  # Only backup if changes were made
        try:
            os.rename(EXISTING_DATA_FILE, BACKUP_DATA_FILE)
            print(f"Backed up original '{EXISTING_DATA_FILE}' to '{BACKUP_DATA_FILE}'.")
        except Exception as e:
            print(f"Warning: Could not create backup of '{EXISTING_DATA_FILE}'. {e}", file=sys.stderr)

    # 6. Save the combined data back to the original file
    if appended_count > 0 or not os.path.exists(
            EXISTING_DATA_FILE):  # Save if new records were added or if original file didn't exist
        save_pickle_data(combined_data, EXISTING_DATA_FILE)
    else:
        print("No new unique records to append. Original file remains unchanged.")


if __name__ == "__main__":
    main()
