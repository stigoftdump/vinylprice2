# /home/stigoftdump/PycharmProjects/PythonProject/vinylprice/checkpickle.py
from persistence import read_ml_data
import pprint  # For pretty printing dictionaries and lists
from collections import Counter, defaultdict
import numpy as np
import pandas as pd  # Added for DataFrame creation for ML model input
import matplotlib.pyplot as plt
from functions import fit_curve_and_get_params  # For the original overall curve plot
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


    # Define a fallback/dummy if import fails, though it won't be as accurate
    def _clean_element_for_feature_name(element_name):
        return str(element_name).lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace(
            '.', '').replace('&', 'and').replace('/', '_').replace('\'', '')

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


# ... (inspect_ml_data, analyze_extra_comments, plot_overall_curve_shape functions remain as they are)
# plot_overall_curve_shape still uses fit_curve_and_get_params for its specific purpose.
# The user's feedback was about the "average album curve shape" visualization.

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


def plot_overall_curve_shape():
    """
    Loads all ML data, fits the sigmoid_plus_exponential curve to it,
    and plots the data points along with the fitted curve.
    THIS FUNCTION USES THE ORIGINAL HARDCODED CURVE FITTING.
    """
    print("\n--- Overall Curve Shape Visualization (Fitted to ALL Data using original curve function) ---")
    sales_list = read_ml_data()
    if not sales_list:
        print("The ML sales data list is empty. Cannot plot curve shape.")
        return

    qualities = []
    prices = []
    for sale_entry in sales_list:
        quality = sale_entry.get('quality')
        price = sale_entry.get('price')
        if quality is not None and price is not None:
            qualities.append(quality)
            prices.append(price)

    if not qualities or len(
            qualities) < MIN_POINTS_FOR_ALBUM_CURVE_FIT:  # Using the constant for this original function
        print(
            f"Not enough data points ({len(qualities)}) with quality and price to fit the original curve. Need at least {MIN_POINTS_FOR_ALBUM_CURVE_FIT}.")
        return

    print(f"Attempting to fit original curve to {len(qualities)} data points...")
    try:
        params, predict_func = fit_curve_and_get_params(qualities, prices)

        if params is None or predict_func is None:
            print("Original curve fitting failed for the overall dataset. Cannot plot.")
            return
        print("Original curve fitting successful for overall data.")

        min_q = min(qualities) if qualities else 0
        max_q = max(qualities) if qualities else 10
        plot_q_min = max(0, min_q - 0.5)
        plot_q_max = min(10, max_q + 0.5)

        quality_range_for_plot = np.linspace(plot_q_min, plot_q_max, 200)
        predicted_prices_for_plot = predict_func(quality_range_for_plot)

        plt.figure(figsize=(12, 7))
        plt.scatter(qualities, prices, label='Actual Sales Data (All)', alpha=0.3, s=15, edgecolor='k', linewidth=0.5)
        plt.plot(quality_range_for_plot, predicted_prices_for_plot, color='red', linewidth=2.5,
                 label='Fitted Original Curve (to All Data)')

        plt.title('Overall Price vs. Quality (Original Curve Fitted to All Data)', fontsize=16)
        plt.xlabel('Quality Score', fontsize=14)
        plt.ylabel('Price (Adjusted)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        min_price = min(prices) if prices else 0
        plot_y_min = 0 if min_price < 50 else max(0, min_price - 10)
        plt.ylim(bottom=plot_y_min)
        plt.xlim(left=0, right=10)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        print("Displaying plot. Close the plot window to continue...")
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib is not installed. Please install it to see the plot: pip install matplotlib")
    except Exception as e:
        print(f"An error occurred during plotting overall original curve: {e}")


def plot_ml_model_average_album_curve_shape(plot_individual_curves=False, plot_outlier_curves=False,
                                            num_albums_to_plot=None):
    """
    Uses the TRAINED ML MODEL to predict price vs. quality for unique albums.
    Normalizes these ML-predicted curves, identifies and excludes outliers,
    then averages the inlier curves and plots the result.
    Optionally plots individual normalized album curves.
    """
    print("\n--- ML Model - Average Normalized Album Curve Shape (Outliers Excluded) ---")

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
            if profile_key[0] and profile_key[1]:
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
    quality_plot_range = np.linspace(0, 9, 100)  # Max actual quality is 9
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

        # --- Optimization: Batch feature creation ---
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
            continue  # Should not happen if quality_plot_range is not empty

        # Create a single DataFrame for all quality points of this album
        df_batch_input = pd.DataFrame(feature_vectors_for_album, columns=ml_feature_names)

        predicted_prices_for_album_profile = []
        try:
            # --- Optimization: Single batch prediction ---
            predicted_prices_for_album_profile = ml_model.predict(df_batch_input)
        except Exception as e:
            # print(f"Warning: Error predicting batch for {profile_key}: {e}")
            # This album profile will be skipped if batch prediction fails
            pass

        if len(predicted_prices_for_album_profile) == len(quality_plot_range):
            max_predicted_price = np.max(predicted_prices_for_album_profile)
            if max_predicted_price > 1e-6:
                normalized_predictions = np.array(predicted_prices_for_album_profile) / max_predicted_price
                all_normalized_ml_predictions.append(normalized_predictions)
                processed_album_count += 1
            else:
                normalization_issues_count_ml += 1
        # else: if prediction failed or returned unexpected length, this album profile is skipped

    if not all_normalized_ml_predictions:
        print(f"No ML-predicted album curves were successfully generated and normalized. Cannot plot average ML curve.")
        if normalization_issues_count_ml > 0:
            print(
                f"{normalization_issues_count_ml} album profiles had issues during normalization (e.g., max ML-predicted price too low).")
        return

    print(f"Successfully generated and normalized ML-predicted curves for {processed_album_count} album profiles.")
    if normalization_issues_count_ml > 0:
        print(f"{normalization_issues_count_ml} album profiles had issues during normalization.")

    # --- Outlier Detection (same logic as before, applied to ML-generated curves) ---
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

        plt.title(f'ML Model - Avg Normalized Album Price vs. Quality ({final_ml_curve_count} inlier profiles)',
                  fontsize=16)
        plt.xlabel('Quality Score (Input to ML Model)', fontsize=14)
        plt.ylabel('Normalized Price (Proportion of Max ML-Predicted Price)', fontsize=14)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=10)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(bottom=-0.05, top=1.05)
        plt.xlim(left=0, right=9)  # Max quality is 9
        plt.xticks(np.arange(0, 9.1, 1), fontsize=12)  # Ticks from 0 to 9
        plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12)

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
    # This plot still uses the original curve fitting, as a baseline or for comparison
    #plot_overall_curve_shape()

    # This is the new function that uses the ML model
    plot_ml_model_average_album_curve_shape(
        plot_individual_curves=True,
        plot_outlier_curves=True,
        num_albums_to_plot=None  # Optional: Limit number of albums to speed up plotting, None for all
    )
