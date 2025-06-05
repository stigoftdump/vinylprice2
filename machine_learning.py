# /home/stigoftdump/PycharmProjects/PythonProject/vinylprice/machine_learning.py
from persistence import read_ml_data
from datetime import datetime  # Import datetime to parse the date string


# import pandas as pd # You might want to use pandas later for more advanced ML tasks

def _get_all_unique_extra_comment_elements(sales_list):
    """
    Scans all sales entries and extracts all unique, stripped elements
    from the 'extra_comments' field.
    """
    all_elements_set = set()
    for sale_entry in sales_list:
        extra_comments_str = sale_entry.get('extra_comments')
        if extra_comments_str:
            elements = [element.strip() for element in extra_comments_str.split(',')]
            for element in elements:
                if element:
                    all_elements_set.add(element)
    return sorted(list(all_elements_set))


def _get_all_unique_artists(sales_list):
    """
    Scans all sales entries and extracts all unique artist names.
    """
    all_artists_set = set()
    for sale_entry in sales_list:
        artist_name = sale_entry.get('artist')
        if artist_name:  # Ensure artist_name is not None or empty
            all_artists_set.add(artist_name.strip())
    return sorted(list(all_artists_set))


def _clean_element_for_feature_name(element_text):
    """
    Cleans a string (from 'extra_comments' or artist name) to create
    a more Pythonic and readable feature name.
    """
    if not isinstance(element_text, str):  # Handle potential non-string inputs
        return f"unknown_val_{abs(hash(element_text)) % 10000}"

    name = element_text.lower()
    name = name.replace(' ', '_').replace('-', '_').replace('.', '').replace('/', '_')
    name = ''.join(char for char in name if char.isalnum() or char == '_')
    name = name.strip('_')
    if name and name[0].isdigit():
        name = f"feat_{name}"
    if not name:
        return f"unknown_element_{abs(hash(element_text)) % 10000}"
    return name


def create_feature_engineered_data_from_list(sales_list):
    """
    Transforms a list of raw sales data into a feature-engineered format.
    Includes:
    - 'quality_score'
    - 'price_adjusted' (target)
    - 'sale_year' (derived from 'date')
    - Boolean features for 'extra_comments' elements.
    - Boolean features for 'artist' (One-Hot Encoded).
    """
    if not sales_list:
        return []

    unique_raw_comment_elements = _get_all_unique_extra_comment_elements(sales_list)
    unique_artists = _get_all_unique_artists(sales_list)

    feature_engineered_list = []

    for sale_entry in sales_list:
        # --- Extract Sale Year ---
        sale_year = None
        date_str = sale_entry.get('date')
        if date_str:
            try:
                # Assuming date is in 'YYYY-MM-DD' format
                sale_year = datetime.strptime(date_str, '%Y-%m-%d').year
            except ValueError:
                print(f"Warning: Could not parse date '{date_str}' for sale entry. Year will be None.")
                # sale_year remains None

        new_entry = {
            'quality_score': sale_entry.get('quality'),
            'price_adjusted': sale_entry.get('price'),
            'sale_year': sale_year,  # Add the extracted year
            # --- Original metadata can be useful for inspection or advanced features later ---
            # 'original_artist': sale_entry.get('artist'),
            # 'original_album': sale_entry.get('album'),
            # 'original_label': sale_entry.get('label'),
            # 'original_native_price': sale_entry.get('native_price'),
            # 'original_date': sale_entry.get('date'),
        }

        # --- Boolean features for 'extra_comments' ---
        current_sale_comment_elements_set = set()
        extra_comments_str = sale_entry.get('extra_comments')
        if extra_comments_str:
            elements = [element.strip() for element in extra_comments_str.split(',')]
            for el in elements:
                if el:
                    current_sale_comment_elements_set.add(el)

        for raw_element in unique_raw_comment_elements:
            feature_name_part = _clean_element_for_feature_name(raw_element)
            feature_name = f"comment_{feature_name_part}"
            new_entry[feature_name] = (raw_element in current_sale_comment_elements_set)

        # --- Boolean features for 'artist' (One-Hot Encoding) ---
        current_artist_name = sale_entry.get('artist')
        if current_artist_name:
            current_artist_name = current_artist_name.strip()

        for artist_from_list in unique_artists:
            # artist_from_list is already stripped from _get_all_unique_artists
            feature_name_part = _clean_element_for_feature_name(artist_from_list)
            feature_name = f"artist_{feature_name_part}"
            new_entry[feature_name] = (current_artist_name == artist_from_list)

        feature_engineered_list.append(new_entry)

    return feature_engineered_list


def generate_features_for_ml_training():
    """
    Reads ML data, generates features, and returns the feature-engineered dataset.
    """
    raw_sales = read_ml_data()
    if not raw_sales:
        print("No raw sales data found in persistence to generate features.")
        return []

    print(f"Starting feature engineering for {len(raw_sales)} raw sales entries...")
    featured_data = create_feature_engineered_data_from_list(raw_sales)
    print(f"Feature engineering complete. Generated {len(featured_data)} featured entries.")
    return featured_data


# --- Example of how you might test this file directly ---
if __name__ == "__main__":
    print("Running machine_learning.py directly for testing...")

    sample_sales_data = [
        {'artist': 'The Velvet Underground', 'album': 'Album 1', 'extra_comments': 'LP, Reissue', 'quality': 8.0,
         'price': 55.00, 'date': '2022-05-10'},
        {'artist': 'Rod Stewart', 'album': 'Album 2', 'extra_comments': 'LP, Gatefold', 'quality': 7.5, 'price': 15.00,
         'date': '2021-11-20'},
        {'artist': 'The Velvet Underground', 'album': 'Album 3', 'extra_comments': 'CD', 'quality': 9.0, 'price': 20.00,
         'date': '2023-01-15'},
        {'artist': 'Unknown Artist', 'album': 'Album 4', 'extra_comments': None, 'quality': 6.0, 'price': 10.00,
         'date': '2020-03-03'},
        {'artist': 'The Beatles', 'album': 'Album 5', 'extra_comments': 'LP, Reissue, 180g', 'quality': 8.5,
         'price': 30.00, 'date': '2022-08-01'},
        {'artist': 'No Date Artist', 'album': 'Album 6', 'extra_comments': 'LP', 'quality': 7.0, 'price': 25.00,
         'date': None},  # Test with no date
    ]

    print("\n--- Testing with sample_sales_data ---")
    featured_data_sample = create_feature_engineered_data_from_list(sample_sales_data)

    if featured_data_sample:
        print(f"Processed {len(featured_data_sample)} sample sales entries.")
        print("\nFeatures for the first sample entry (The Velvet Underground):")
        for key, value in featured_data_sample[0].items():
            if key.startswith('artist_') or key in ['quality_score', 'price_adjusted', 'sale_year'] or key.startswith(
                    'comment_'):
                print(f"  {key}: {value}")

        print("\nFeatures for the second sample entry (Rod Stewart):")
        for key, value in featured_data_sample[1].items():
            if key.startswith('artist_') or key in ['quality_score', 'price_adjusted', 'sale_year'] or key.startswith(
                    'comment_'):
                print(f"  {key}: {value}")

        print("\nFeatures for the entry with no date:")
        for key, value in featured_data_sample[5].items():
            if key.startswith('artist_') or key in ['quality_score', 'price_adjusted', 'sale_year'] or key.startswith(
                    'comment_'):
                print(f"  {key}: {value}")
    else:
        print("No featured data generated from sample.")

    print("\n--- Testing with data from ml_data_pkl (if it exists) ---")
    featured_data_from_file = generate_features_for_ml_training()
    if featured_data_from_file:
        print(f"\nSuccessfully processed {len(featured_data_from_file)} sales entries from ml_data_pkl.")

        # Determine how many entries to show, up to 6
        num_entries_to_show = min(len(featured_data_from_file), 6)

        if num_entries_to_show > 0:
            print(
                f"\nExample features for the first {num_entries_to_show} entries from ml_data_pkl (showing artist, comment, quality, price, year):")
            for i in range(num_entries_to_show):
                entry_to_show = featured_data_from_file[i]
                print(f"\n--- Entry {i + 1} ---")
                features_shown_count = 0
                for key, value in entry_to_show.items():
                    if key.startswith('artist_') or key.startswith('comment_') or \
                            key in ['quality_score', 'price_adjusted', 'sale_year']:
                        # Show True flags, or core values (quality, price, year)
                        if value is True or key in ['quality_score', 'price_adjusted', 'sale_year']:
                            print(f"  {key}: {value}")
                            features_shown_count += 1

                # If very few relevant features were true/present, you might want to show more.
                # This part can be adjusted based on how much detail you want.
                if features_shown_count < 5 and len(entry_to_show) > 5:
                    print("  (Showing a few more general features as few specific flags were True):")
                    count = 0
                    for key, value in entry_to_show.items():
                        # Avoid re-printing what was already shown if possible, or just show a sample
                        if not (key.startswith('artist_') or key.startswith('comment_') or \
                                key in ['quality_score', 'price_adjusted', 'sale_year']):
                            if value is not False:  # Avoid printing lots of False flags
                                print(f"    {key}: {value}")
                                count += 1
                                if count >= 3:  # Show a few extra general ones
                                    break
            if len(featured_data_from_file) > num_entries_to_show:
                print(
                    f"\n... and {len(featured_data_from_file) - num_entries_to_show} more processed entries in the file.")
        else:
            print("No processed entries from ml_data_pkl to display.")

    else:
        print("No featured data generated from ml_data_pkl (it might be empty or not found).")
