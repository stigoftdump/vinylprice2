# /home/stigoftdump/PycharmProjects/PythonProject/vinylprice/checkpickle.py
from persistence import read_ml_data
import pprint  # For pretty printing dictionaries and lists
from collections import Counter


def inspect_ml_data():
    sales_list = read_ml_data()

    print("--- ML Data Inspection (Unique Records Summary) ---")

    if not sales_list:
        print("The ML sales data list is empty.")
        return

    print(f"Total individual sales entries in ML data: {len(sales_list)}")

    # Collect unique records first
    unique_record_identifiers = set()
    for sale_entry in sales_list:
        album = sale_entry.get('album', 'N/A')
        artist = sale_entry.get('artist', 'N/A')
        extra_comments = sale_entry.get('extra_comments', 'N/A')
        # Store as (artist, album, extra_comments) for easier sorting later
        record_identifier = (artist, album, extra_comments)
        unique_record_identifiers.add(record_identifier)

    print(f"Total unique records (Artist/Album/Extra Comments): {len(unique_record_identifiers)}\n")

    print("Summary of unique records (Artist, Album, Extra Comments):")

    # Define column widths for the table
    artist_col_width = 30
    album_col_width = 40
    extra_comments_col_header = "Extra Comments"

    # Print table header - Artist first, then Album
    header = (f"{'Artist':<{artist_col_width}} | "
              f"{'Album':<{album_col_width}} | "
              f"{extra_comments_col_header}")
    print(header)
    print("-" * (artist_col_width + album_col_width + len(extra_comments_col_header) + 6))  # +6 for " | " separators

    # Convert set of tuples to a list and sort it
    # Sort by Artist (index 0), then Album (index 1), then Extra Comments (index 2)
    sorted_unique_records = sorted(list(unique_record_identifiers),
                                   key=lambda x: (x[0].lower(), x[1].lower(), x[2].lower()))

    for artist, album, extra_comments in sorted_unique_records:
        # Truncate fields if they are too long for their columns
        artist_display = (artist[:artist_col_width - 3] + "...") if len(artist) > artist_col_width else artist
        album_display = (album[:album_col_width - 3] + "...") if len(album) > album_col_width else album

        print(f"{artist_display:<{artist_col_width}} | "
              f"{album_display:<{album_col_width}} | "
              f"{extra_comments}")


def analyze_extra_comments():
    """
    Analyzes the 'extra_comments' field from the ML data to count
    occurrences of each element and provide an example including
    Artist, Album, and the full extra_comments string, displayed in a table.
    """
    sales_list = read_ml_data()
    if not sales_list:
        print("\n--- Extra Comments Analysis ---")
        print("The ML sales data list is empty. No comments to analyze.")
        return

    all_elements_counter = Counter()
    # Store a tuple: (full_extra_comments, artist, album)
    first_example_details_for_element = {}

    for sale_entry in sales_list:
        extra_comments_str = sale_entry.get('extra_comments')
        artist_name = sale_entry.get('artist', 'N/A')
        album_name = sale_entry.get('album', 'N/A')

        if extra_comments_str:  # Check if the string is not None or empty
            # Split the string by ", " and strip whitespace from each part
            elements = [element.strip() for element in extra_comments_str.split(',')]
            for element in elements:
                if element:  # Ensure the element itself is not an empty string after stripping
                    all_elements_counter[element] += 1
                    if element not in first_example_details_for_element:
                        first_example_details_for_element[element] = (extra_comments_str, artist_name, album_name)

    print("\n--- Extra Comments Analysis ---")
    if not all_elements_counter:
        print("No 'extra_comments' found or all were empty.")
        return

    print(f"Found {len(all_elements_counter)} unique elements in 'extra_comments'.\n")

    # Define column widths
    element_col_width = 30
    count_col_width = 10
    # Example column will take the rest, but let's define a minimum for header
    example_col_header = "First Example Context"

    # Print table header
    header = (f"{'Element':<{element_col_width}} | "
              f"{'Count':>{count_col_width}} | "
              f"{example_col_header}")
    print(header)
    print("-" * len(header))

    # Sort elements alphabetically by element name (the key of the counter item)
    # Convert counter items to a list of tuples and sort by the first element (element name)
    sorted_elements = sorted(all_elements_counter.items(), key=lambda item: item[0].lower())

    for element, count in sorted_elements:
        example_details = first_example_details_for_element.get(element)
        if example_details:
            full_extra_comments, artist, album = example_details
            example_context = f"Artist: '{artist}', Album: '{album}', Extra Comments: '{full_extra_comments}'"
        else:
            example_context = "N/A"

        # Truncate element if it's too long for its column, or adjust column width
        element_display = (element[:element_col_width - 3] + "...") if len(element) > element_col_width else element

        print(f"{element_display:<{element_col_width}} | "
              f"{count:>{count_col_width}} | "
              f"{example_context}")


if __name__ == "__main__":
    inspect_ml_data()
    analyze_extra_comments()
