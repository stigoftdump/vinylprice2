from persistence import read_ml_data
from collections import Counter, defaultdict
import numpy as np
import pandas as pd  # Added for DataFrame creation for ML model input
import matplotlib.pyplot as plt
import pickle  # For loading the ML model
import os  # For path joining
from datetime import datetime

# Attempt to import _clean_element_for_feature_name from machine_learning.py
# This is needed to map artist/comment strings to the feature names the model expects.
try:
    from machine_learning import _clean_element_for_feature_name
except ImportError:
    print("Warning: Could not import _clean_element_for_feature_name from machine_learning.py.")
    print("ML model curve plotting might not correctly map artist/comment features.")

# --- Constants ---
MIN_POINTS_FOR_ALBUM_CURVE_FIT = 7  # For the original curve fitting function
OUTLIER_STD_FACTOR = 2.0
DEFAULT_SALE_YEAR_FOR_ML_PLOT = 2023  # Fallback year if not found for an album

# Define paths for ML model and features (mirroring train_global_model.py)
BASE_DIR_CKP = os.path.dirname(os.path.abspath(__file__))  # ckp for checkpickle
GLOBAL_MODEL_FILENAME_CKP = "global_price_model.pkl"
MODEL_FEATURES_FILENAME_CKP = "model_features.pkl"
GLOBAL_MODEL_PATH_CKP = os.path.join(BASE_DIR_CKP, GLOBAL_MODEL_FILENAME_CKP)
MODEL_FEATURES_PATH_CKP = os.path.join(BASE_DIR_CKP, MODEL_FEATURES_FILENAME_CKP)

def inspect_ml_data():
    """
    Inspects the ML data stored in the new dictionary format.
    Displays a summary of releases and a sample of their sales, including all saved fields.
    """
    all_releases_data = read_ml_data()  # This now returns a dictionary

    print("--- ML Data Inspection (New Dictionary Format) ---")

    if not all_releases_data:
        print("The ML releases data dictionary is empty.")
        return

    print(f"Total unique releases in ML data: {len(all_releases_data)}\n")

    # --- Release Overview ---
    print("--- Release Overview ---")
    # Define column widths for the release overview table
    ro_id_w = 12
    ro_artist_w = 30
    ro_title_w = 40
    ro_year_w = 6
    ro_sales_count_w = 10

    header_ro = (f"{'Discogs ID':<{ro_id_w}} | {'API Artist':<{ro_artist_w}} | {'API Title':<{ro_title_w}} | "
                 f"{'Year':<{ro_year_w}} | {'# Sales':>{ro_sales_count_w}}")
    print(header_ro)
    print("-" * (ro_id_w + ro_artist_w + ro_title_w + ro_year_w + ro_sales_count_w + 10))

    # Sort releases by ID for consistent display
    sorted_release_ids = sorted(all_releases_data.keys(), key=lambda x: int(x) if x.isdigit() else x)

    for release_id in sorted_release_ids:
        release_entry = all_releases_data[release_id]
        artist = str(release_entry.get('api_artist', 'N/A'))
        title = str(release_entry.get('api_title', 'N/A'))
        year = str(release_entry.get('api_year', 'N/A'))
        sales_history = release_entry.get('sales_history', [])
        sales_count = len(sales_history)

        artist_display = (artist[:ro_artist_w - 3] + "...") if len(artist) > ro_artist_w else artist
        title_display = (title[:ro_title_w - 3] + "...") if len(title) > ro_title_w else title

        print(f"{str(release_id):<{ro_id_w}} | {artist_display:<{ro_artist_w}} | {title_display:<{ro_title_w}} | "
              f"{year:<{ro_year_w}} | {sales_count:>{ro_sales_count_w}}")
    print("--- End of Release Overview ---\n")

    # --- Detailed Sample of Releases and Their Sales (First 5 Releases, First 5 Sales Each) ---
    print("--- Detailed Sample of Releases and Their Sales ---")
    releases_to_sample = sorted_release_ids[:5]  # Sample first 5 releases by ID

    if not releases_to_sample:
        print("No releases to sample in detail.")
    else:
        for i, release_id in enumerate(releases_to_sample):
            release_entry = all_releases_data[release_id]
            print(f"\nRelease Sample #{i + 1}: ID = {release_id}")
            print(f"  API Artist: {release_entry.get('api_artist', 'N/A')}")
            print(f"  API Title:  {release_entry.get('api_title', 'N/A')}")
            print(f"  API Year:   {release_entry.get('api_year', 'N/A')}")
            print(f"  API Country: {release_entry.get('api_country', 'N/A')}")
            print(f"  API Label:  {release_entry.get('api_label', 'N/A')}")
            print(f"  API Cat#:   {release_entry.get('api_catno', 'N/A')}")

            genres_list = release_entry.get('api_genres', [])
            print(f"  API Genres: {', '.join(genres_list) if genres_list else 'N/A'}")

            styles_list = release_entry.get('api_styles', [])
            print(f"  API Styles: {', '.join(styles_list) if styles_list else 'N/A'}")

            formats_list = release_entry.get('api_format_descriptions', [])
            print(f"  API Formats: {', '.join(formats_list) if formats_list else 'N/A'}")

            notes = str(release_entry.get('api_notes', 'N/A'))
            notes_display = (notes[:70] + "...") if len(notes) > 73 else notes  # Truncate long notes
            print(f"  API Notes:  {notes_display}")

            print(f"  API Community Have: {release_entry.get('api_community_have', 'N/A')}")
            print(f"  API Community Want: {release_entry.get('api_community_want', 'N/A')}")
            avg_rating = release_entry.get('api_community_rating_average', 'N/A')
            rating_count = release_entry.get('api_community_rating_count', 'N/A')
            print(f"  API Community Rating: {avg_rating} (from {rating_count} ratings)")

            sales_history = release_entry.get('sales_history', [])
            if not sales_history:
                print("  No sales history for this release.")
            else:
                print(f"  Sales History (first 5 of {len(sales_history)}):")
                # Define column widths for the sales table
                s_date_w = 10
                s_qual_w = 8
                s_price_w = 10
                s_native_w = 12
                s_comment_w = 40

                sales_header = (f"    {'Date':<{s_date_w}} | {'Quality':<{s_qual_w}} | {'Price Adj.':<{s_price_w}} | "
                                f"{'Native Price':<{s_native_w}} | {'Sale Comment':<{s_comment_w}}")
                print(sales_header)
                print("    " + "-" * (s_date_w + s_qual_w + s_price_w + s_native_w + s_comment_w + 10))

                for sale_idx, sale_data in enumerate(sales_history[:5]):
                    date_val = sale_data.get('date', 'N/A')
                    quality = sale_data.get('quality', 'N/A')
                    quality_str = f"{quality:.2f}" if isinstance(quality, (float, np.float64)) else str(quality)
                    price = sale_data.get('price', 'N/A')  # This is inflation-adjusted
                    price_str = f"{price:.2f}" if isinstance(price, (float, np.float64)) else str(price)
                    native_price = sale_data.get('native_price', 'N/A')
                    sale_comment = str(sale_data.get('sale_comment', 'N/A'))
                    comment_display = (sale_comment[:s_comment_w - 3] + "...") if len(
                        sale_comment) > s_comment_w else sale_comment

                    print(f"    {str(date_val):<{s_date_w}} | {quality_str:<{s_qual_w}} | {price_str:<{s_price_w}} | "
                          f"{str(native_price):<{s_native_w}} | {comment_display:<{s_comment_w}}")
    print("--- End of Detailed Sample ---")

def analyze_extra_comments():  # Consider renaming to analyze_api_format_descriptions
    """
    Analyzes the 'api_format_descriptions' field from the ML data to count
    occurrences of each description element and provide an example context.
    """
    all_releases_data = read_ml_data()  # This is a dictionary of releases
    if not all_releases_data:
        print("\n--- API Format Descriptions Analysis ---")
        print("The ML releases data dictionary is empty. No descriptions to analyze.")
        return

    all_elements_counter = Counter()
    # Store a tuple: (full_list_of_descriptions_for_release, api_artist, api_title)
    first_example_details_for_element = {}

    for release_id, release_entry in all_releases_data.items():
        # 'api_format_descriptions' is expected to be a list of strings
        format_descriptions_list = release_entry.get('api_format_descriptions', [])
        api_artist = release_entry.get('api_artist', 'N/A')
        api_title = release_entry.get('api_title', 'N/A')

        if format_descriptions_list:  # Check if the list is not None or empty
            for element in format_descriptions_list:  # Each element is a description string
                if element and element.strip():  # Ensure the element itself is not an empty string
                    stripped_element = element.strip()
                    all_elements_counter[stripped_element] += 1
                    if stripped_element not in first_example_details_for_element:
                        # Store the original list and context for the first example
                        first_example_details_for_element[stripped_element] = (
                            format_descriptions_list, api_artist, api_title
                        )

    print("\n--- API Format Descriptions Analysis ---")  # Updated title
    if not all_elements_counter:
        print("No 'api_format_descriptions' found or all were empty.")
        return

    print(f"Found {len(all_elements_counter)} unique elements in 'api_format_descriptions'.\n")

    # Define column widths
    element_col_width = 30
    count_col_width = 10
    example_col_header = "First Example Context"

    # Print table header
    header = (f"{'Description Element':<{element_col_width}} | "  # Updated label
              f"{'Count':>{count_col_width}} | "
              f"{example_col_header}")
    print(header)
    print("-" * (len(header) + 5))  # Adjusted separator length

    # Sort elements alphabetically
    sorted_elements = sorted(all_elements_counter.items(), key=lambda item: item[0].lower())

    for element, count in sorted_elements:
        example_details = first_example_details_for_element.get(element)
        if example_details:
            full_descriptions_list, artist, title = example_details
            # Join the list of descriptions for display
            descriptions_str_for_example = ", ".join(full_descriptions_list)
            example_context = f"Artist: '{artist}', Title: '{title}', Format Descriptions: '{descriptions_str_for_example}'"
        else:
            example_context = "N/A"

        element_display = (element[:element_col_width - 3] + "...") if len(element) > element_col_width else element

        print(f"{element_display:<{element_col_width}} | "
              f"{count:>{count_col_width}} | "
              f"{example_context}")

def plot_ml_model_average_album_curve_shape(plot_individual_curves=False, plot_outlier_curves=False,
                                            num_albums_to_plot=None):
    """
    Uses the TRAINED ML MODEL to predict price vs. quality for unique albums.
    Normalizes these ML-predicted curves by the price at quality score 6,
    identifies and excludes outliers, then averages the inlier curves and plots the result.
    Optionally plots individual normalized album curves.
    """
    print("\n--- ML Model - Average Normalized Album Curve Shape (Normalized by Price at Q6, Outliers Excluded) ---")

    # 1. Load ML Model and Features
    ml_model = None
    ml_feature_names = None
    try:
        with open(GLOBAL_MODEL_PATH_CKP, 'rb') as f:
            ml_model = pickle.load(f)
        with open(MODEL_FEATURES_PATH_CKP, 'rb') as f:
            ml_feature_names = pickle.load(f)
        if not ml_model or not ml_feature_names:
            raise FileNotFoundError("Model or features are None after loading.")
        print(f"Successfully loaded ML model and {len(ml_feature_names)} feature names.")
    except FileNotFoundError:
        print(f"Error: ML Model ({GLOBAL_MODEL_PATH_CKP}) or Features ({MODEL_FEATURES_PATH_CKP}) not found.")
        return
    except Exception as e:
        print(f"Error loading ML model or features: {e}")
        return

    # 2. Load Raw Sales Data
    sales_list = read_ml_data()
    if not sales_list:
        print("The ML sales data list is empty. Cannot plot ML model album curve shape.")
        return

    # 3. Group sales by unique album to get profiles
    unique_album_profiles = defaultdict(list)
    for sale_entry in sales_list:
        artist = sale_entry.get('artist')
        album_name = sale_entry.get('album')
        extra_comments = sale_entry.get('extra_comments', '')

        if artist and album_name:
            profile_key = (str(artist).strip(), str(album_name).strip(), str(extra_comments).strip())
            if profile_key[0] and profile_key[1]:  # Ensure artist and album are not empty
                unique_album_profiles[profile_key].append(sale_entry)

    if not unique_album_profiles:
        print("No valid unique album profiles found for ML curve generation.")
        return

    print(f"Found {len(unique_album_profiles)} unique album profiles from sales data.")

    if num_albums_to_plot and num_albums_to_plot < len(unique_album_profiles):
        print(f"Plotting for a sample of {num_albums_to_plot} albums.")
        profile_items_to_process_list = list(unique_album_profiles.items())[:num_albums_to_plot]
    else:
        profile_items_to_process_list = list(unique_album_profiles.items())

    total_profiles_to_process = len(profile_items_to_process_list)
    if total_profiles_to_process == 0:
        print("No album profiles to process.")
        return

    all_normalized_ml_predictions = []
    quality_plot_range = np.linspace(0, 9, 100)  # Max actual quality is 9. This range includes 6.0.
    # Find the index corresponding to quality score 6
    # This assumes 6.0 is representable in quality_plot_range.
    # If quality_plot_range changes, this might need adjustment or use np.isclose.
    try:
        idx_q6 = np.where(np.isclose(quality_plot_range, 6.0))[0][0]
    except IndexError:
        print("Error: Quality score 6.0 not found in quality_plot_range. Check np.linspace parameters.")
        # Fallback or error handling if 6.0 isn't exactly in the range
        # For robustness, find the closest index:
        idx_q6 = np.abs(quality_plot_range - 6.0).argmin()
        print(f"Using closest quality score to 6.0: {quality_plot_range[idx_q6]:.2f}")

    processed_album_count = 0
    normalization_issues_count_ml = 0
    current_profile_index = 0

    for profile_key, album_sales_entries in profile_items_to_process_list:
        current_profile_index += 1
        artist_val, album_val, comments_val_str = profile_key
        print(
            f"Processing album profile {current_profile_index} of {total_profiles_to_process}: {artist_val} - {album_val[:30]}...")

        sale_years_for_album = []
        for s_entry in album_sales_entries:
            date_val = s_entry.get('date')
            if date_val:
                if isinstance(date_val, str):
                    try:
                        dt_obj = datetime.strptime(date_val, '%Y-%m-%d')
                        sale_years_for_album.append(dt_obj.year)
                    except ValueError:
                        print(
                            f"Warning: Could not parse date string '{date_val}' for album {profile_key}. Skipping this date.")
                elif hasattr(date_val, 'year'):
                    sale_years_for_album.append(date_val.year)

        if sale_years_for_album:
            representative_sale_year = int(np.median(sale_years_for_album))
        else:
            representative_sale_year = DEFAULT_SALE_YEAR_FOR_ML_PLOT

        feature_vectors_for_album = []
        for q_score in quality_plot_range:
            current_features = {feat: 0 for feat in ml_feature_names}
            current_features['quality_score'] = q_score
            if 'sale_year' in current_features:
                current_features['sale_year'] = representative_sale_year

            artist_feat_name = f"artist_{_clean_element_for_feature_name(artist_val)}"
            if artist_feat_name in current_features:
                current_features[artist_feat_name] = 1

            if comments_val_str:
                comment_elements = [c.strip() for c in comments_val_str.split(',') if c.strip()]
                for comment_element in comment_elements:
                    comment_feat_name = f"comment_{_clean_element_for_feature_name(comment_element)}"
                    if comment_feat_name in current_features:
                        current_features[comment_feat_name] = 1
            feature_vectors_for_album.append(current_features)

        if not feature_vectors_for_album:
            continue

        df_batch_input = pd.DataFrame(feature_vectors_for_album, columns=ml_feature_names)
        predicted_prices_for_album_profile = []
        try:
            predicted_prices_for_album_profile = ml_model.predict(df_batch_input)
        except Exception as e:
            # print(f"Warning: Error predicting batch for {profile_key}: {e}")
            pass

        if len(predicted_prices_for_album_profile) == len(quality_plot_range):
            price_at_quality_6 = predicted_prices_for_album_profile[idx_q6]

            if price_at_quality_6 > 1e-6:  # Avoid division by zero or near-zero
                normalized_predictions = np.array(predicted_prices_for_album_profile) / price_at_quality_6
                all_normalized_ml_predictions.append(normalized_predictions)
                processed_album_count += 1
            else:
                # print(f"Warning: Price at Q6 for {profile_key} is too low ({price_at_quality_6:.2f}) for normalization. Skipping.")
                normalization_issues_count_ml += 1
        # else: if prediction failed or returned unexpected length, this album profile is skipped

    if not all_normalized_ml_predictions:
        print(f"No ML-predicted album curves were successfully generated and normalized. Cannot plot average ML curve.")
        if normalization_issues_count_ml > 0:
            print(
                f"{normalization_issues_count_ml} album profiles had issues during normalization (e.g., price at Q6 too low).")
        return

    print(f"Successfully generated and normalized ML-predicted curves for {processed_album_count} album profiles.")
    if normalization_issues_count_ml > 0:
        print(f"{normalization_issues_count_ml} album profiles had issues during normalization (price at Q6 too low).")

    # --- Outlier Detection ---
    if processed_album_count < 3:
        print("Too few ML-predicted curves to perform robust outlier detection. Averaging all.")
        inlier_ml_predictions = all_normalized_ml_predictions
        outlier_ml_predictions = []
    else:
        predictions_matrix_ml_all = np.array(all_normalized_ml_predictions)
        provisional_average_ml = np.mean(predictions_matrix_ml_all, axis=0)

        deviations_rmse_ml = []
        for norm_curve_ml in predictions_matrix_ml_all:
            rmse_ml = np.sqrt(np.mean((norm_curve_ml - provisional_average_ml) ** 2))
            deviations_rmse_ml.append(rmse_ml)

        mean_rmse_ml = np.mean(deviations_rmse_ml)
        std_rmse_ml = np.std(deviations_rmse_ml)
        outlier_threshold_ml = mean_rmse_ml + (OUTLIER_STD_FACTOR * std_rmse_ml)

        print(
            f"ML Outlier detection: Mean RMSE = {mean_rmse_ml:.4f}, Std RMSE = {std_rmse_ml:.4f}, Threshold = {outlier_threshold_ml:.4f}")

        inlier_ml_predictions = []
        outlier_ml_predictions = []
        for i, norm_curve_ml in enumerate(all_normalized_ml_predictions):
            if deviations_rmse_ml[i] <= outlier_threshold_ml:
                inlier_ml_predictions.append(norm_curve_ml)
            else:
                outlier_ml_predictions.append(norm_curve_ml)
        print(f"Identified and excluded {len(outlier_ml_predictions)} outlier ML-predicted curve shapes.")

    if not inlier_ml_predictions:
        print("No inlier ML-predicted curves remaining after outlier detection.")
        return

    final_average_ml_normalized_prices = np.mean(np.array(inlier_ml_predictions), axis=0)
    final_ml_curve_count = len(inlier_ml_predictions)

    print(f"Averaging {final_ml_curve_count} inlier ML-predicted normalized curves for the final plot.")

    # --- Plotting ---
    try:
        plt.figure(figsize=(12, 7))

        if plot_individual_curves:
            for i, ml_preds in enumerate(inlier_ml_predictions):
                plt.plot(quality_plot_range, ml_preds, color='lightskyblue', linewidth=0.7, alpha=0.25,
                         label='Individual Inlier ML Fits' if i == 0 else None)

        if plot_outlier_curves and outlier_ml_predictions:
            for i, ml_preds in enumerate(outlier_ml_predictions):
                plt.plot(quality_plot_range, ml_preds, color='lightcoral', linewidth=0.7, alpha=0.35,
                         label='Individual Outlier ML Fits' if i == 0 else None)

        plt.plot(quality_plot_range, final_average_ml_normalized_prices, color='dodgerblue', linewidth=3,
                 label=f'Average Normalized ML Curve Shape ({final_ml_curve_count} Inlier Album Profiles)')

        title_suffix = f'Normalized by Price at Q6 ({final_ml_curve_count} inlier profiles)'
        plt.title(f'ML Model - Avg Album Price vs. Quality ({title_suffix})',
                  fontsize=16)
        plt.xlabel('Quality Score (Input to ML Model)', fontsize=14)
        plt.ylabel('Normalized Price (Proportion of Price at Quality 6)', fontsize=14)  # Updated Y-axis label

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=10)

        plt.grid(True, linestyle='--', alpha=0.7)
        # Adjust Y-axis limits if necessary, as values can now be > 1 or < 0 more easily
        # For example, if price at Q9 is 2x price at Q6, value is 2.0.
        # If price at Q3 is 0.5x price at Q6, value is 0.5.
        # The price at Q6 itself will be 1.0.
        # A good general range might be based on the min/max of final_average_ml_normalized_prices
        min_norm_val = np.min(final_average_ml_normalized_prices) if final_average_ml_normalized_prices.size > 0 else 0
        max_norm_val = np.max(
            final_average_ml_normalized_prices) if final_average_ml_normalized_prices.size > 0 else 1.2
        plt.ylim(bottom=min(0, min_norm_val - 0.1), top=max(1.1, max_norm_val + 0.1))

        plt.xlim(left=0, right=9)  # Max quality is 9
        plt.xticks(np.arange(0, 9.1, 1), fontsize=12)  # Ticks from 0 to 9
        # Y-ticks might need to be more dynamic too, but let's start with this:
        plt.yticks(np.arange(round(min(0, min_norm_val - 0.1), 1), round(max(1.1, max_norm_val + 0.1), 1) + 0.1, 0.2),
                   fontsize=12)

        print("Displaying ML model's average normalized album curve plot. Close the plot window to continue...")
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib is not installed. Please install it to see the plot: pip install matplotlib")
    except Exception as e:
        print(f"An error occurred during plotting ML model's average normalized album curve: {e}")

if __name__ == "__main__":
    inspect_ml_data()
    analyze_extra_comments()

    # This is the new function that uses the ML model
    #plot_ml_model_average_album_curve_shape(plot_individual_curves=True, plot_outlier_curves=True, num_albums_to_plot=None)
