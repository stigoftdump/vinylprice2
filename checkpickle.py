# /home/stigoftdump/PycharmProjects/PythonProject/vinylprice/checkpickle.py
from persistence import read_ml_data
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sys  # Added for stderr
from datetime import datetime

try:
    from machine_learning import _clean_element_for_feature_name
except ImportError:
    print("Warning: Could not import _clean_element_for_feature_name from machine_learning.py.", file=sys.stderr)
    print("ML model curve plotting might not correctly map artist/comment features.", file=sys.stderr)


    def _clean_element_for_feature_name(element_name):  # Fallback
        return str(element_name).lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace(
            '.', '').replace('&', 'and').replace('/', '_').replace('\'', '')

# --- Constants ---
MIN_POINTS_FOR_ALBUM_CURVE_FIT = 7
OUTLIER_STD_FACTOR = 2.0
DEFAULT_SALE_YEAR_FOR_ML_PLOT = 2023

BASE_DIR_CKP = os.path.dirname(os.path.abspath(__file__))
GLOBAL_MODEL_FILENAME_CKP = "global_price_model.pkl"
MODEL_FEATURES_FILENAME_CKP = "model_features.pkl"
GLOBAL_MODEL_PATH_CKP = os.path.join(BASE_DIR_CKP, GLOBAL_MODEL_FILENAME_CKP)
MODEL_FEATURES_PATH_CKP = os.path.join(BASE_DIR_CKP, MODEL_FEATURES_FILENAME_CKP)


def inspect_ml_data():
    """
    Inspects the ML data (dictionary of releases) stored in ml_data.pkl.
    Displays a summary of releases and a sample of their API data and sales history.
    """
    all_releases_data = read_ml_data()

    print("--- ML Data Inspection (New Dictionary Format) ---")
    if not all_releases_data:
        print("The ML releases data dictionary is empty.")
        return
    print(f"Total unique releases in ML data: {len(all_releases_data)}\n")

    print("--- Release Overview ---")
    ro_id_w, ro_artist_w, ro_title_w, ro_year_w, ro_sales_count_w = 12, 30, 40, 6, 10
    header_ro = (f"{'Discogs ID':<{ro_id_w}} | {'API Artist':<{ro_artist_w}} | {'API Title':<{ro_title_w}} | "
                 f"{'Year':<{ro_year_w}} | {'# Sales':>{ro_sales_count_w}}")
    print(header_ro)
    print("-" * (ro_id_w + ro_artist_w + ro_title_w + ro_year_w + ro_sales_count_w + 10))

    sorted_release_ids = sorted(all_releases_data.keys(), key=lambda x: int(x) if x.isdigit() else x)

    for release_id in sorted_release_ids:
        release_entry = all_releases_data[release_id]
        artist = str(release_entry.get('api_artist', 'N/A'))
        title = str(release_entry.get('api_title', 'N/A'))
        # Use api_original_year for overview if available, fallback to api_year
        year_to_display = release_entry.get('api_original_year') or release_entry.get('api_year', 'N/A')
        year = str(year_to_display)
        sales_count = len(release_entry.get('sales_history', []))
        artist_display = (artist[:ro_artist_w - 3] + "...") if len(artist) > ro_artist_w else artist
        title_display = (title[:ro_title_w - 3] + "...") if len(title) > ro_title_w else title
        print(f"{str(release_id):<{ro_id_w}} | {artist_display:<{ro_artist_w}} | {title_display:<{ro_title_w}} | "
              f"{year:<{ro_year_w}} | {sales_count:>{ro_sales_count_w}}")
    print("--- End of Release Overview ---\n")

    print("--- Detailed Sample of Releases and Their Sales ---")
    releases_to_sample = sorted_release_ids[:5]

    if not releases_to_sample:
        print("No releases to sample in detail.")
    else:
        for i, release_id in enumerate(releases_to_sample):
            release_entry = all_releases_data[release_id]
            print(f"\nRelease Sample #{i + 1}: ID = {release_id}")
            print(f"  API Master ID: {release_entry.get('api_master_id', 'N/A')}")
            print(f"  API Artist: {release_entry.get('api_artist', 'N/A')}")
            print(f"  API Title:  {release_entry.get('api_title', 'N/A')}")
            print(f"  API Original Year: {release_entry.get('api_original_year', 'N/A')}")
            print(f"  API Year (Pressing): {release_entry.get('api_year', 'N/A')}")  # Clarified label
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
            notes_display = (notes[:70] + "...") if len(notes) > 73 else notes
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
                s_date_w, s_qual_w, s_price_w, s_native_w, s_comment_w = 10, 8, 10, 12, 40
                sales_header = (f"    {'Date':<{s_date_w}} | {'Quality':<{s_qual_w}} | {'Price Adj.':<{s_price_w}} | "
                                f"{'Native Price':<{s_native_w}} | {'Sale Comment':<{s_comment_w}}")
                print(sales_header)
                print("    " + "-" * (s_date_w + s_qual_w + s_price_w + s_native_w + s_comment_w + 10))
                for sale_data in sales_history[:5]:
                    date_val = sale_data.get('date', 'N/A')
                    quality = sale_data.get('quality', 'N/A')
                    quality_str = f"{quality:.2f}" if isinstance(quality, (float, np.float64)) else str(quality)
                    price = sale_data.get('price', 'N/A')
                    price_str = f"{price:.2f}" if isinstance(price, (float, np.float64)) else str(price)
                    native_price = sale_data.get('native_price', 'N/A')
                    sale_comment = str(sale_data.get('sale_comment', 'N/A'))
                    comment_display = (sale_comment[:s_comment_w - 3] + "...") if len(
                        sale_comment) > s_comment_w else sale_comment
                    print(f"    {str(date_val):<{s_date_w}} | {quality_str:<{s_qual_w}} | {price_str:<{s_price_w}} | "
                          f"{str(native_price):<{s_native_w}} | {comment_display:<{s_comment_w}}")
    print("--- End of Detailed Sample ---")


def analyze_extra_comments():
    """
    Analyzes the 'api_format_descriptions' field from the ML data to count
    occurrences of each description element and provide an example context.
    """
    all_releases_data = read_ml_data()
    if not all_releases_data:
        print("\n--- API Format Descriptions Analysis ---")
        print("The ML releases data dictionary is empty. No descriptions to analyze.")
        return

    all_elements_counter = Counter()
    first_example_details_for_element = {}

    for release_id, release_entry in all_releases_data.items():
        format_descriptions_list = release_entry.get('api_format_descriptions', [])
        api_artist = release_entry.get('api_artist', 'N/A')
        api_title = release_entry.get('api_title', 'N/A')

        if format_descriptions_list:
            for element in format_descriptions_list:
                if element and element.strip():
                    stripped_element = element.strip()
                    all_elements_counter[stripped_element] += 1
                    if stripped_element not in first_example_details_for_element:
                        first_example_details_for_element[stripped_element] = (
                            format_descriptions_list, api_artist, api_title
                        )

    print("\n--- API Format Descriptions Analysis ---")
    if not all_elements_counter:
        print("No 'api_format_descriptions' found or all were empty.")
        return
    print(f"Found {len(all_elements_counter)} unique elements in 'api_format_descriptions'.\n")

    element_col_width, count_col_width = 30, 10
    example_col_header = "First Example Context"
    header = (f"{'Description Element':<{element_col_width}} | {'Count':>{count_col_width}} | {example_col_header}")
    print(header)
    print("-" * (len(header) + 5))

    sorted_elements = sorted(all_elements_counter.items(), key=lambda item: item[0].lower())
    for element, count in sorted_elements:
        example_details = first_example_details_for_element.get(element)
        if example_details:
            full_list, artist, title = example_details
            example_context = f"Artist: '{artist}', Title: '{title}', Formats: '{', '.join(full_list)}'"
        else:
            example_context = "N/A"
        element_display = (element[:element_col_width - 3] + "...") if len(element) > element_col_width else element
        print(f"{element_display:<{element_col_width}} | {count:>{count_col_width}} | {example_context}")


def plot_ml_model_average_album_curve_shape(plot_individual_curves=False, plot_outlier_curves=False,
                                            num_albums_to_plot=None):
    """
    Uses the TRAINED ML MODEL to predict price vs. quality for unique releases.
    It processes each release from the `ml_data.pkl` (dictionary format).
    For each release, it uses its API metadata and sales history to generate
    feature vectors for a range of quality scores. The ML model predicts prices
    for these scores. These predicted price curves are normalized by the
    ML-predicted price at quality score 6. Outliers among these normalized
    curves are identified and excluded. The remaining inlier curves are averaged
    and plotted. Optionally plots individual normalized album curves.
    """
    print(
        "\n--- ML Model - Average Normalized Release Curve Shape (Normalized by ML Price at Q6, Outliers Excluded) ---")

    ml_model, ml_feature_names = None, None
    try:
        with open(GLOBAL_MODEL_PATH_CKP, 'rb') as f:
            ml_model = pickle.load(f)
        with open(MODEL_FEATURES_PATH_CKP, 'rb') as f:
            ml_feature_names = pickle.load(f)
        if not ml_model or not ml_feature_names: raise FileNotFoundError("Model or features are None.")
        print(f"Successfully loaded ML model and {len(ml_feature_names)} feature names.")
    except FileNotFoundError:
        print(f"Error: ML Model ({GLOBAL_MODEL_PATH_CKP}) or Features ({MODEL_FEATURES_PATH_CKP}) not found.",
              file=sys.stderr)
        return
    except Exception as e:
        print(f"Error loading ML model or features: {e}", file=sys.stderr)
        return

    all_releases_data = read_ml_data()
    if not all_releases_data:
        print("The ML releases data dictionary is empty. Cannot plot ML model curve shape.")
        return

    print(f"Processing {len(all_releases_data)} unique releases from ML data.")

    # Determine which releases to process (all or a sample)
    release_ids_to_process = list(all_releases_data.keys())
    if num_albums_to_plot and num_albums_to_plot < len(release_ids_to_process):
        print(f"Plotting for a sample of {num_albums_to_plot} releases.")
        # Potentially sort or sample strategically if needed, for now just take the first N
        release_ids_to_process = release_ids_to_process[:num_albums_to_plot]

    total_releases_to_process = len(release_ids_to_process)
    if total_releases_to_process == 0:
        print("No releases to process.")
        return

    all_normalized_ml_predictions = []
    quality_plot_range = np.linspace(0, 9, 100)
    try:
        idx_q6 = np.where(np.isclose(quality_plot_range, 6.0))[0][0]
    except IndexError:
        idx_q6 = np.abs(quality_plot_range - 6.0).argmin()
        print(f"Warning: Quality score 6.0 not found in plot range. Using closest: {quality_plot_range[idx_q6]:.2f}",
              file=sys.stderr)

    processed_release_count = 0
    normalization_issues_count_ml = 0

    for current_idx, release_id in enumerate(release_ids_to_process):
        release_entry = all_releases_data[release_id]
        api_artist = release_entry.get('api_artist', 'Unknown Artist')
        api_title = release_entry.get('api_title', 'Unknown Title')
        # Use original year if available, then pressing year, then default
        representative_sale_year = release_entry.get('api_original_year') or \
                                   release_entry.get('api_year') or \
                                   DEFAULT_SALE_YEAR_FOR_ML_PLOT
        try:  # Ensure year is int
            representative_sale_year = int(representative_sale_year)
        except (ValueError, TypeError):
            representative_sale_year = DEFAULT_SALE_YEAR_FOR_ML_PLOT
            print(f"Warning: Could not parse year for {release_id}, using default {DEFAULT_SALE_YEAR_FOR_ML_PLOT}",
                  file=sys.stderr)

        # Use API format descriptions as "comments" for feature generation
        format_descriptions = release_entry.get('api_format_descriptions', [])

        print(
            f"Processing release {current_idx + 1} of {total_releases_to_process}: ID {release_id} ({api_artist} - {api_title[:30]})")

        feature_vectors_for_release = []
        for q_score in quality_plot_range:
            current_features = {feat: 0 for feat in ml_feature_names}  # Reset for each quality score
            current_features['quality_score'] = q_score
            if 'sale_year' in current_features:  # Check if 'sale_year' is an expected feature
                current_features['sale_year'] = representative_sale_year

            # Artist feature
            artist_feat_name = f"artist_{_clean_element_for_feature_name(api_artist)}"
            if artist_feat_name in current_features:
                current_features[artist_feat_name] = 1

            # Format description features (acting as 'comments')
            for desc in format_descriptions:
                format_feat_name = f"comment_{_clean_element_for_feature_name(desc)}"  # Using 'comment_' prefix
                if format_feat_name in current_features:
                    current_features[format_feat_name] = 1

            # Add other API-based features if they are part of ml_feature_names
            # Example: Genre features
            for genre in release_entry.get('api_genres', []):
                genre_feat_name = f"genre_{_clean_element_for_feature_name(genre)}"
                if genre_feat_name in current_features:
                    current_features[genre_feat_name] = 1
            # Example: Style features
            for style in release_entry.get('api_styles', []):
                style_feat_name = f"style_{_clean_element_for_feature_name(style)}"
                if style_feat_name in current_features:
                    current_features[style_feat_name] = 1
            # Example: Country feature
            country = release_entry.get('api_country')
            if country:
                country_feat_name = f"country_{_clean_element_for_feature_name(country)}"
                if country_feat_name in current_features:
                    current_features[country_feat_name] = 1

            feature_vectors_for_release.append(current_features)

        if not feature_vectors_for_release:
            continue

        df_batch_input = pd.DataFrame(feature_vectors_for_release, columns=ml_feature_names).fillna(
            0)  # Fill NaNs with 0
        predicted_prices_for_release = []
        try:
            predicted_prices_for_release = ml_model.predict(df_batch_input)
        except Exception as e:
            print(f"Warning: Error predicting batch for release ID {release_id}: {e}", file=sys.stderr)
            continue  # Skip this release if prediction fails

        if len(predicted_prices_for_release) == len(quality_plot_range):
            price_at_quality_6 = predicted_prices_for_release[idx_q6]
            if price_at_quality_6 > 1e-6:
                normalized_predictions = np.array(predicted_prices_for_release) / price_at_quality_6
                all_normalized_ml_predictions.append(normalized_predictions)
                processed_release_count += 1
            else:
                normalization_issues_count_ml += 1
        # else: prediction failed or returned unexpected length

    if not all_normalized_ml_predictions:
        print(f"No ML-predicted release curves were successfully generated/normalized. Cannot plot average ML curve.")
        if normalization_issues_count_ml > 0:
            print(
                f"{normalization_issues_count_ml} releases had issues during normalization (e.g., price at Q6 too low).")
        return

    print(f"Successfully generated and normalized ML-predicted curves for {processed_release_count} releases.")
    if normalization_issues_count_ml > 0:
        print(f"{normalization_issues_count_ml} releases had issues during normalization (price at Q6 too low).")

    # --- Outlier Detection ---
    if processed_release_count < 3:
        print("Too few ML-predicted curves for robust outlier detection. Averaging all.")
        inlier_ml_predictions = all_normalized_ml_predictions
        outlier_ml_predictions = []
    else:
        predictions_matrix_ml_all = np.array(all_normalized_ml_predictions)
        provisional_average_ml = np.mean(predictions_matrix_ml_all, axis=0)
        deviations_rmse_ml = [np.sqrt(np.mean((curve - provisional_average_ml) ** 2)) for curve in
                              predictions_matrix_ml_all]
        mean_rmse_ml, std_rmse_ml = np.mean(deviations_rmse_ml), np.std(deviations_rmse_ml)
        outlier_threshold_ml = mean_rmse_ml + (OUTLIER_STD_FACTOR * std_rmse_ml)
        print(
            f"ML Outlier detection: Mean RMSE={mean_rmse_ml:.4f}, Std RMSE={std_rmse_ml:.4f}, Threshold={outlier_threshold_ml:.4f}")
        inlier_ml_predictions = [curve for i, curve in enumerate(all_normalized_ml_predictions) if
                                 deviations_rmse_ml[i] <= outlier_threshold_ml]
        outlier_ml_predictions = [curve for i, curve in enumerate(all_normalized_ml_predictions) if
                                  deviations_rmse_ml[i] > outlier_threshold_ml]
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
                 label=f'Average Normalized ML Curve ({final_ml_curve_count} Inlier Releases)')
        title_suffix = f'Normalized by ML Price at Q6 ({final_ml_curve_count} inlier releases)'
        plt.title(f'ML Model - Avg Release Price vs. Quality ({title_suffix})', fontsize=16)
        plt.xlabel('Quality Score (Input to ML Model)', fontsize=14)
        plt.ylabel('Normalized Price (Proportion of Price at Quality 6)', fontsize=14)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        min_norm_val = np.min(final_average_ml_normalized_prices) if final_average_ml_normalized_prices.size > 0 else 0
        max_norm_val = np.max(
            final_average_ml_normalized_prices) if final_average_ml_normalized_prices.size > 0 else 1.2
        plt.ylim(bottom=min(0, min_norm_val - 0.1), top=max(1.1, max_norm_val + 0.1))
        plt.xlim(left=0, right=9)
        plt.xticks(np.arange(0, 9.1, 1), fontsize=12)
        plt.yticks(np.arange(round(min(0, min_norm_val - 0.1), 1), round(max(1.1, max_norm_val + 0.1), 1) + 0.1, 0.2),
                   fontsize=12)
        print("Displaying ML model's average normalized release curve plot. Close the plot window to continue...")
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib is not installed. Please install it to see the plot: pip install matplotlib", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred during plotting ML model's average normalized release curve: {e}", file=sys.stderr)


if __name__ == "__main__":
    inspect_ml_data()
    analyze_extra_comments()
    # Ensure you have a trained model and features file for this to work.
    #plot_ml_model_average_album_curve_shape(plot_individual_curves=True, plot_outlier_curves=True, num_albums_to_plot=100) # Example: plot for 100 albums
