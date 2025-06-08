# /home/stigoftdump/PycharmProjects/PythonProject/vinylprice/machine_learning.py
from scripts.persistence import read_ml_data
from datetime import datetime
import sys  # For printing warnings/errors


# _clean_element_for_feature_name remains largely the same,
# but let's ensure it's robust and consistent.
def _clean_element_for_feature_name(element_text):
    """
    Cleans a string to create a Pythonic and readable feature name.
    Handles various characters and ensures the name is a valid identifier.
    """
    if not isinstance(element_text, str):
        # Create a somewhat predictable name for non-string inputs
        return f"unknown_val_{abs(hash(str(element_text))) % 10000}"

    name = element_text.lower()
    # Replace common separators and problematic characters
    name = name.replace(' ', '_').replace('-', '_').replace('.', '').replace('/', '_')
    name = name.replace('(', '').replace(')', '').replace('&', 'and').replace('\'', '')
    # Keep only alphanumeric characters and underscores
    name = ''.join(char for char in name if char.isalnum() or char == '_')
    # Remove leading/trailing underscores that might result from cleaning
    name = name.strip('_')

    # Ensure feature name doesn't start with a digit
    if name and name[0].isdigit():
        name = f"feat_{name}"
    # Handle cases where cleaning results in an empty string
    if not name:
        return f"unknown_element_{abs(hash(element_text)) % 10000}"
    return name


# --- New Helper Functions to get unique elements across all releases ---

def _get_all_unique_api_elements(all_releases_data, api_key, is_list_of_strings=False):
    """
    Generic helper to extract unique, cleaned elements from a specific API key
    across all release entries.
    """
    all_elements_set = set()
    if not all_releases_data:
        return []
    for release_id, release_entry in all_releases_data.items():
        value = release_entry.get(api_key)
        if value:
            if is_list_of_strings:  # e.g., api_genres, api_styles, api_format_descriptions
                for item_str in value:
                    if item_str and isinstance(item_str, str):
                        all_elements_set.add(item_str.strip())
            elif isinstance(value, str):  # e.g., api_artist, api_country
                all_elements_set.add(value.strip())
            # Add handling for other types if necessary, e.g. int for year if we were to one-hot encode it
    return sorted(list(all_elements_set))


def create_feature_engineered_data_from_releases(all_releases_data):
    """
    Transforms the dictionary of releases (from ml_data.pkl) into a
    feature-engineered list of sales, suitable for ML model training.

    Each sale from each release becomes a row, enriched with features
    derived from both the sale itself and the release's API metadata.
    """
    if not all_releases_data:
        print("Warning: No release data provided to feature engineering.", file=sys.stderr)
        return []

    # Get all unique elements that will become one-hot encoded features
    unique_api_artists = _get_all_unique_api_elements(all_releases_data, 'api_artist')
    unique_api_genres = _get_all_unique_api_elements(all_releases_data, 'api_genres', is_list_of_strings=True)
    unique_api_styles = _get_all_unique_api_elements(all_releases_data, 'api_styles', is_list_of_strings=True)
    unique_api_countries = _get_all_unique_api_elements(all_releases_data, 'api_country')
    # Using 'api_format_descriptions' for 'comment_' features, consistent with checkpickle.py plotting
    unique_api_format_descriptions = _get_all_unique_api_elements(all_releases_data, 'api_format_descriptions',
                                                                  is_list_of_strings=True)

    feature_engineered_list = []
    processed_sales_count = 0

    for release_id, release_entry in all_releases_data.items():
        sales_history = release_entry.get('sales_history', [])

        # Extract release-level API data once per release
        api_artist = release_entry.get('api_artist')
        api_original_year = release_entry.get('api_original_year')
        # Fallback to pressing year if original year is not available
        if api_original_year is None:
            api_original_year = release_entry.get('api_year')
        try:  # Ensure year is an integer
            api_original_year = int(api_original_year) if api_original_year is not None else None
        except (ValueError, TypeError):
            print(f"Warning: Could not parse api_original_year for release {release_id}. Setting to None.",
                  file=sys.stderr)
            api_original_year = None

        api_genres = release_entry.get('api_genres', [])
        api_styles = release_entry.get('api_styles', [])
        api_country = release_entry.get('api_country')
        api_format_descriptions = release_entry.get('api_format_descriptions', [])

        for sale_data in sales_history:
            new_entry = {}

            # --- Core sale-specific features ---
            new_entry['quality_score'] = sale_data.get('quality')
            new_entry['price_adjusted'] = sale_data.get('price')  # This is the target variable

            sale_year = None
            date_str = sale_data.get('date')
            if date_str:
                try:
                    sale_year = datetime.strptime(date_str, '%Y-%m-%d').year
                except ValueError:
                    print(
                        f"Warning: Could not parse date '{date_str}' for sale in release {release_id}. Sale year will be None.",
                        file=sys.stderr)
            new_entry['sale_year'] = sale_year

            # --- Release-level API features ---
            new_entry['api_original_year'] = api_original_year  # Direct numerical feature

            # One-hot encode API Artist
            cleaned_api_artist_name = _clean_element_for_feature_name(api_artist) if api_artist else None
            for unique_artist_val in unique_api_artists:
                feat_name = f"artist_{_clean_element_for_feature_name(unique_artist_val)}"
                new_entry[feat_name] = (cleaned_api_artist_name == _clean_element_for_feature_name(unique_artist_val))

            # One-hot encode API Genres
            current_release_genres_cleaned = {_clean_element_for_feature_name(g) for g in api_genres if g}
            for unique_genre_val in unique_api_genres:
                feat_name = f"genre_{_clean_element_for_feature_name(unique_genre_val)}"
                new_entry[feat_name] = (
                            _clean_element_for_feature_name(unique_genre_val) in current_release_genres_cleaned)

            # One-hot encode API Styles
            current_release_styles_cleaned = {_clean_element_for_feature_name(s) for s in api_styles if s}
            for unique_style_val in unique_api_styles:
                feat_name = f"style_{_clean_element_for_feature_name(unique_style_val)}"
                new_entry[feat_name] = (
                            _clean_element_for_feature_name(unique_style_val) in current_release_styles_cleaned)

            # One-hot encode API Country
            cleaned_api_country_name = _clean_element_for_feature_name(api_country) if api_country else None
            for unique_country_val in unique_api_countries:
                feat_name = f"country_{_clean_element_for_feature_name(unique_country_val)}"
                new_entry[feat_name] = (cleaned_api_country_name == _clean_element_for_feature_name(unique_country_val))

            # One-hot encode API Format Descriptions (as 'comment_' features)
            current_release_formats_cleaned = {_clean_element_for_feature_name(f) for f in api_format_descriptions if f}
            for unique_format_val in unique_api_format_descriptions:
                feat_name = f"comment_{_clean_element_for_feature_name(unique_format_val)}"  # Using 'comment_' prefix
                new_entry[feat_name] = (
                            _clean_element_for_feature_name(unique_format_val) in current_release_formats_cleaned)

            # --- Add other potential features here ---
            # e.g., community ratings, have/want counts from release_entry if desired

            feature_engineered_list.append(new_entry)
            processed_sales_count += 1

    print(
        f"Feature engineering complete. Generated {len(feature_engineered_list)} featured entries from {processed_sales_count} sales across {len(all_releases_data)} releases.")
    return feature_engineered_list


def generate_features_for_ml_training():
    """
    Reads ML data (dictionary of releases), generates features for each sale,
    and returns the feature-engineered dataset.
    """
    all_releases_data = read_ml_data()
    if not all_releases_data:
        print("No ML data found in persistence to generate features.", file=sys.stderr)
        return []

    print(f"Starting feature engineering for {len(all_releases_data)} releases...")
    featured_data = create_feature_engineered_data_from_releases(all_releases_data)
    return featured_data


# --- Example of how you might test this file directly ---
if __name__ == "__main__":
    print("Running machine_learning.py directly for testing...")

    # Sample data mimicking the structure of ml_data.pkl
    sample_ml_data = {
        "12345": {  # discogs_release_id
            "api_artist": "The Future Sound of London",
            "api_title": "Lifeforms",
            "api_original_year": 1994,
            "api_year": 1994,  # Pressing year
            "api_genres": ["Electronic"],
            "api_styles": ["Ambient", "IDM"],
            "api_country": "UK",
            "api_format_descriptions": ["2xLP", "Album", "Gatefold"],
            "sales_history": [
                {'date': '2022-05-10', 'quality': 8.0, 'price': 55.00, 'native_price': "£45.00",
                 'sale_comment': "NM/NM"},
                {'date': '2021-11-20', 'quality': 7.5, 'price': 48.00, 'native_price': "£40.00",
                 'sale_comment': "VG+/VG+"},
            ]
        },
        "67890": {
            "api_artist": "Massive Attack",
            "api_title": "Mezzanine",
            "api_original_year": 1998,
            "api_year": 2018,  # A reissue
            "api_genres": ["Electronic", "Trip Hop"],
            "api_styles": ["Downtempo", "Trip Hop"],
            "api_country": "Europe",
            "api_format_descriptions": ["2xLP", "Album", "Reissue", "180g"],
            "sales_history": [
                {'date': '2023-01-15', 'quality': 9.0, 'price': 30.00, 'native_price': "€28.00",
                 'sale_comment': "Sealed"},
                {'date': '2022-08-01', 'quality': 8.5, 'price': 25.00, 'native_price': "€23.00",
                 'sale_comment': "Opened but unplayed"},
            ]
        },
        "11223": {  # Release with fewer API details and one sale
            "api_artist": "Aphex Twin",
            "api_title": "Selected Ambient Works Volume II",
            "api_original_year": 1994,
            "api_year": 1994,
            "api_genres": ["Electronic"],
            "api_styles": ["Ambient"],
            # api_country missing
            # api_format_descriptions missing
            "sales_history": [
                {'date': '2020-03-03', 'quality': 6.0, 'price': 70.00, 'native_price': "$80.00",
                 'sale_comment': "VG, plays well"},
            ]
        }
    }

    print("\n--- Testing with sample_ml_data (mimicking ml_data.pkl structure) ---")
    featured_data_sample = create_feature_engineered_data_from_releases(sample_ml_data)

    if featured_data_sample:
        print(f"Processed {len(featured_data_sample)} sales entries from sample_ml_data.")

        # Display features for the first few processed sales
        for i, entry in enumerate(featured_data_sample[:3]):  # Show first 3 sales
            print(f"\n--- Features for Sample Sale Entry #{i + 1} ---")
            # Identify which release this sale belongs to for context
            original_release_id = "Unknown"
            if entry.get('api_original_year') == 1994 and 'artist_the_future_sound_of_london' in entry and entry[
                'artist_the_future_sound_of_london']:
                original_release_id = "12345 (FSOL)"
            elif entry.get('api_original_year') == 1998 and 'artist_massive_attack' in entry and entry[
                'artist_massive_attack']:
                original_release_id = "67890 (Massive Attack)"
            elif entry.get('api_original_year') == 1994 and 'artist_aphex_twin' in entry and entry['artist_aphex_twin']:
                original_release_id = "11223 (Aphex Twin)"

            print(f"  (Derived from Release ID context: {original_release_id})")
            print(f"  quality_score: {entry.get('quality_score')}")
            print(f"  price_adjusted (target): {entry.get('price_adjusted')}")
            print(f"  sale_year: {entry.get('sale_year')}")
            print(f"  api_original_year: {entry.get('api_original_year')}")

            print("  One-hot features (showing TRUE flags):")
            for key, value in entry.items():
                if (key.startswith('artist_') or \
                    key.startswith('genre_') or \
                    key.startswith('style_') or \
                    key.startswith('country_') or \
                    key.startswith('comment_')) and value is True:
                    print(f"    {key}: {value}")
            if i == 2 and len(featured_data_sample) > 3:
                print(f"... and {len(featured_data_sample) - 3} more sales processed from sample.")
                break
    else:
        print("No featured data generated from sample_ml_data.")

    print("\n--- Testing with data from actual ml_data.pkl (if it exists) ---")
    featured_data_from_file = generate_features_for_ml_training()
    if featured_data_from_file:
        print(f"\nSuccessfully processed {len(featured_data_from_file)} sales entries from ml_data.pkl.")
        # You can add more detailed printouts for the file data if needed, similar to the sample.
    else:
        print("No featured data generated from ml_data.pkl (it might be empty or not found).")
