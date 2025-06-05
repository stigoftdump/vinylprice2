# check_ml_data.py
from persistence import read_ml_data
import pprint # For pretty printing dictionaries and lists

def inspect_ml_data():
    sales_list, last_record_id = read_ml_data()

    print("--- ML Data Inspection ---")
    print(f"Last Record ID: {last_record_id}\n")

    if not sales_list:
        print("The ML sales data list is empty.")
        return

    print(f"Total sales entries: {len(sales_list)}\n")

    print("First few sales entries (up to 3):") # Showing up to 3 to keep it concise
    for i, sale_entry in enumerate(sales_list[:90]): # Iterate through the first 3 entries
        print(f"\nEntry {i+1}:")
        # Pretty print the dictionary to see all fields clearly
        pprint.pprint(sale_entry, indent=2, width=100) # Added indent and width for readability

    # If you want to see more, you can loop through all or a larger slice:
    # print("\nAll sales entries:")
    # for i, sale_entry in enumerate(sales_list):
    #     print(f"\nEntry {i+1}:")
    #     pprint.pprint(sale_entry, indent=2, width=100)

    # Check for consistency in record_ids
    record_ids_found = set()
    for sale_entry in sales_list:
        record_ids_found.add(sale_entry.get('record_id'))

    print(f"\nUnique Record IDs found in sales data: {sorted(list(record_ids_found))}")
    if record_ids_found and max(record_ids_found) != last_record_id:
        print(f"Warning: Max record_id in sales ({max(record_ids_found)}) does not match last_record_id ({last_record_id})")
    elif not record_ids_found and last_record_id != 0:
        print(f"Warning: No record_ids in sales data, but last_record_id is {last_record_id}")
    elif record_ids_found and max(record_ids_found) == last_record_id:
        print("Info: Max record_id in sales matches last_record_id.")


if __name__ == "__main__":
    inspect_ml_data()