from persistence import read_ml_data
import pprint # For pretty printing dictionaries and lists

def inspect_ml_data():
    sales_list = read_ml_data() # read_ml_data now only returns the list

    print("--- ML Data Inspection ---")

    if not sales_list:
        print("The ML sales data list is empty.")
        return

    print(f"Total sales entries: {len(sales_list)}\n")

    print("First few sales entries (up to 3):")
    for i, sale_entry in enumerate(sales_list[:56]):
        print(f"\nEntry {i+1}:")
        pprint.pprint(sale_entry, indent=2, width=100)

    # Example: Check for a specific field if needed
    # artists_found = set()
    # for sale_entry in sales_list:
    #     artists_found.add(sale_entry.get('artist'))
    # print(f"\nUnique Artists found: {artists_found}")

if __name__ == "__main__":
    inspect_ml_data()