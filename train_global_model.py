# /home/stigoftdump/PycharmProjects/PythonProject/vinylprice/train_global_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import numpy as np  # For np.sqrt
import sys  # For printing to stderr
import os  # For path joining
from collections import defaultdict  # For grouping records
from datetime import datetime  # For evaluate_model_on_training_data
from sklearn.model_selection import KFold, cross_val_score

# Assuming machine_learning.py is in the same directory or accessible in PYTHONPATH
from machine_learning import (
    generate_features_for_ml_training,
    create_feature_engineered_data_from_list,  # Used by generate_features_for_ml_training
    _clean_element_for_feature_name,  # Used by evaluate_model_on_training_data
    _get_all_unique_extra_comment_elements,  # Used by evaluate_model_on_training_data
    _get_all_unique_artists  # Used by evaluate_model_on_training_data
)

# Import functions from functions.py and persistence.py
try:
    from functions import fit_curve_and_get_params, predict_price as predict_price_local
    from persistence import read_ml_data
except ImportError as e:
    print(f"CRITICAL: Could not import necessary modules: {e}", file=sys.stderr)
    print("Please ensure functions.py and persistence.py are accessible.", file=sys.stderr)


    # Define dummy functions to allow the script to run partially
    def fit_curve_and_get_params(*args, **kwargs):
        print("Dummy fit_curve_and_get_params called due to import error.", file=sys.stderr)
        return None, None


    def predict_price_local(*args, **kwargs):
        print("Dummy predict_price_local called due to import error.", file=sys.stderr)
        return 0.0


    def read_ml_data():
        print("Dummy read_ml_data called due to import error.", file=sys.stderr)
        return []

MODEL_FILENAME = "global_price_model.pkl"
FEATURES_FILENAME = "model_features.pkl"

# Define paths relative to the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GLOBAL_MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)
MODEL_FEATURES_PATH = os.path.join(BASE_DIR, FEATURES_FILENAME)

# sets the random state - set to None for truly random, or 42 to be consistent.
random_state = 42
MIN_POINTS_PER_RECORD_FOR_FIT = 7  # Minimum sales for a record to be evaluated

def train_and_evaluate_model():
    """
    Loads feature-engineered data, trains a RandomForestRegressor model,
    evaluates it using cross-validation, and saves the model (trained on 80% of data)
    and its feature list.
    """
    print("Starting model training process...")

    # 1. Load Feature-Engineered Data
    featured_data_list = generate_features_for_ml_training()

    if not featured_data_list:
        print("No feature-engineered data available. Aborting training.", file=sys.stderr)
        return

    # 2. Convert to Pandas DataFrame
    df = pd.DataFrame(featured_data_list)
    print(f"Converted data to DataFrame. Shape: {df.shape}")

    # 3. Prepare X (features) and y (target) - using the full processed data
    if 'price_adjusted' not in df.columns:
        print("Error: 'price_adjusted' (target variable) not found in the DataFrame.", file=sys.stderr)
        return

    y = df['price_adjusted']
    X = df.drop('price_adjusted', axis=1)

    # --- Sanity check for target variable ---
    if y.isnull().any():
        print(
            f"Warning: Target variable 'price_adjusted' contains {y.isnull().sum()} missing values. These rows will be dropped.",
            file=sys.stderr)
        valid_indices = y.dropna().index
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]
        if X.empty or y.empty: # Check both X and y
            print("Error: No valid data remaining after dropping rows with missing target values.", file=sys.stderr)
            return
        print(f"Shape after dropping rows with missing target: X={X.shape}, y={y.shape}")

    # 4. Handle Missing Values in X (specifically for 'sale_year')
    median_year = 0
    if 'sale_year' in X.columns:
        if X['sale_year'].isnull().any():
            median_year_calc = X['sale_year'].median() # Calculate median on the current X
            if pd.isna(median_year_calc):
                print("Warning: All 'sale_year' values are missing in the dataset. Filling with 0.",
                      file=sys.stderr)
                median_year = 0
            else:
                median_year = int(median_year_calc)
            print(f"Filling {X['sale_year'].isnull().sum()} missing 'sale_year' values with median: {median_year}")
            X['sale_year'] = X['sale_year'].fillna(median_year)
    else:
        print("Warning: 'sale_year' column not found in features.", file=sys.stderr)

    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)

    non_numeric_cols = X.select_dtypes(exclude=np.number).columns
    if len(non_numeric_cols) > 0:
        print(f"Error: Non-numeric columns found in features after pre-processing: {list(non_numeric_cols)}",
              file=sys.stderr)
        print("Please ensure all features are numeric or properly encoded.", file=sys.stderr)
        return

    if X.empty:
        print("Error: Feature set X is empty. Cannot proceed with training.", file=sys.stderr)
        return

    print(f"Final feature set shape for evaluation: {X.shape}")

    # 5. Initialize Model (same as before)
    model_for_cv = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=False, n_jobs=-1)
    # Note: oob_score for the model used in cross_val_score is not strictly necessary
    # as CV provides its own out-of-sample estimates.
    # We'll use a separate model instance for the final training with OOB.
    print("RandomForestRegressor model initialized for Cross-Validation.")

    # --- Cross-Validation Evaluation ---
    print("\n--- Performing Cross-Validation Evaluation ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    try:
        # Use a list to store scores for different metrics if needed
        cv_neg_mse_scores = cross_val_score(model_for_cv, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
        cv_r2_scores = cross_val_score(model_for_cv, X, y, cv=kf, scoring='r2', n_jobs=-1)

        cv_mse_scores = -cv_neg_mse_scores
        cv_rmse_scores = np.sqrt(cv_mse_scores)

        print(f"Cross-Validation MSE scores for each fold: {cv_mse_scores.round(2)}")
        print(f"Cross-Validation RMSE scores for each fold: {cv_rmse_scores.round(2)}")
        print(f"Cross-Validation R2 scores for each fold: {cv_r2_scores.round(4)}")

        print("\n--- Average Cross-Validation Metrics ---")
        print(f"Average MSE: {np.mean(cv_mse_scores):.2f} (Std: {np.std(cv_mse_scores):.2f})")
        print(f"Average RMSE: {np.mean(cv_rmse_scores):.2f} (Std: {np.std(cv_rmse_scores):.2f})")
        print(f"Average R2 Score: {np.mean(cv_r2_scores):.4f} (Std: {np.std(cv_r2_scores):.4f})")
        print("----------------------------------------")

    except Exception as e:
        print(f"Error during cross-validation: {e}", file=sys.stderr)
        print("Skipping cross-validation due to error.", file=sys.stderr)
        # If CV fails, we might not want to proceed or just note it.
        # For now, we'll continue to train and save a model on a single split.

    # --- Train final model on 80% of data and save (for actual use) ---
    print("\n--- Training final model on 80% data for saving ---")
    # This split is reproducible due to random_state=random_state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Use a new model instance for training the final model to get OOB score
    final_model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True, n_jobs=-1)
    final_model.fit(X_train, y_train)
    print("Final model training complete.")

    if hasattr(final_model, 'oob_score_') and final_model.oob_score_:
        print(f"Model Out-of-Bag (OOB) R^2 score (from final model): {final_model.oob_score_:.4f}")

    # --- Evaluate the saved model on the specific 20% test set (Optional but good for comparison) ---
    print("\n--- Global ML Model Evaluation on Specific Test Set (20%) ---")
    y_pred_test = final_model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)

    print(f"Mean Squared Error (MSE): {mse_test:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse_test:.2f}")
    print(f"R-squared (R2 Score): {r2_test:.4f}")
    print("------------------------------------")

    # 10. Save the Model and Feature List (using the model trained on 80%)
    try:
        with open(GLOBAL_MODEL_PATH, 'wb') as f:
            pickle.dump(final_model, f) # Save the final_model
        print(f"Trained model saved to {GLOBAL_MODEL_PATH}")

        feature_names = list(X_train.columns) # Features from the training set of the final model
        with open(MODEL_FEATURES_PATH, 'wb') as f:
            pickle.dump(feature_names, f)
        print(f"Model feature list saved to {MODEL_FEATURES_PATH}")

    except Exception as e:
        print(f"Error saving model or features: {e}", file=sys.stderr)

    # --- Call the other evaluation functions ---
    evaluate_per_record_local_curve_fitting()
    evaluate_model_on_training_data() # This evaluates the *saved* (final_model) on the full data

def evaluate_model_on_training_data():
    """
    Loads the saved global ML model and features, applies it to the entire ML training data,
    and reports evaluation metrics on the training set.
    """
    print("\n--- Evaluating Global ML Model on Full Training Data ---")
    model = None
    feature_names = None  # These are the features the model was TRAINED on
    try:
        with open(GLOBAL_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(MODEL_FEATURES_PATH, 'rb') as f:
            feature_names = pickle.load(f)
        print("Loaded saved global ML model and feature names.")
    except FileNotFoundError:
        print(
            f"Error: Saved global ML model ({GLOBAL_MODEL_PATH}) or features ({MODEL_FEATURES_PATH}) not found. Cannot evaluate on training data.",
            file=sys.stderr)
        return
    except Exception as e:
        print(f"Error loading saved global ML model or features: {e}. Cannot evaluate on training data.",
              file=sys.stderr)
        return

    if model is None or feature_names is None:
        return

    raw_sales = read_ml_data()
    if not raw_sales:
        print("No raw sales data found in persistence to evaluate on.", file=sys.stderr)
        return
    print(f"Loaded {len(raw_sales)} raw sales entries for training data evaluation.")

    # 1. Use the canonical feature engineering function from machine_learning.py
    # This function returns a list of dictionaries, each containing all
    # engineered features AND the 'price_adjusted' target.
    all_features_and_target_list = create_feature_engineered_data_from_list(raw_sales)

    if not all_features_and_target_list:
        print(
            "Feature engineering for evaluation data (using create_feature_engineered_data_from_list) yielded no results.",
            file=sys.stderr)
        return

    df_eval_full = pd.DataFrame(all_features_and_target_list)

    # 2. Separate target (y) and handle missing target values
    if 'price_adjusted' not in df_eval_full.columns:
        print("Error: 'price_adjusted' (target) not found in DataFrame after feature engineering for evaluation.",
              file=sys.stderr)
        return

    y_train_actual = df_eval_full['price_adjusted']

    valid_indices_eval = y_train_actual.dropna().index  # Get indices where target is not NaN

    # Check if all target values were NaN initially
    if valid_indices_eval.empty and not y_train_actual.empty:
        print(
            "Error: All 'price_adjusted' values were NaN in the evaluation data. This is unexpected if tester.py shows valid prices.",
            file=sys.stderr)
        return

    y_train_actual = y_train_actual.loc[valid_indices_eval]  # Filter y to valid targets

    # If, after dropping NaNs from y, there's no data left
    if y_train_actual.empty and len(all_features_and_target_list) > 0:  # Check original list length
        print("Error: No valid data remaining for evaluation after dropping missing targets from 'price_adjusted'.",
              file=sys.stderr)
        return

    # 3. Prepare features (X) using only the columns listed in `feature_names`
    #    and filter X based on valid target indices.
    #    Ensure columns are in the order expected by the model.

    # Create an empty DataFrame with the expected feature columns
    X_train_eval = pd.DataFrame(columns=feature_names)

    # Populate X_train_eval from df_eval_full, only for rows with valid targets
    # and only with the features the model was trained on.
    for feat_name in feature_names:
        if feat_name in df_eval_full.columns:
            X_train_eval[feat_name] = df_eval_full.loc[valid_indices_eval, feat_name]
        else:
            # This case means a feature the model was trained on is not present
            # in the data generated by create_feature_engineered_data_from_list.
            # This could happen if, for example, a very rare comment was present during
            # training but not in the current full raw_sales set (unlikely if raw_sales is the same).
            # For evaluation on the full training set, this should ideally not occur.
            # If it does, filling with 0 is a common strategy.
            print(
                f"Warning: Feature '{feat_name}' (from training) not found in current feature engineered data. Filling with 0 for evaluation.",
                file=sys.stderr)
            X_train_eval[feat_name] = 0

            # Ensure X_train_eval is not empty after selection and potential NaN drops from y
    if X_train_eval.empty and not y_train_actual.empty:  # If y_train_actual has data but X_train_eval became empty
        print("Error: X_train_eval became empty after filtering based on valid targets or feature selection.",
              file=sys.stderr)
        return
    elif X_train_eval.empty and y_train_actual.empty and len(all_features_and_target_list) > 0:
        # This is the original error condition path
        print("Error: No valid data remaining for evaluation after dropping missing targets (X_train_eval is empty).",
              file=sys.stderr)
        return

    # 4. Handle Missing Values in X_train_eval (e.g., 'sale_year')
    # This should be consistent with how it was handled during training.
    # create_feature_engineered_data_from_list already handles sale_year extraction.
    # Imputation for sale_year should happen *after* X_train_eval is constructed with the correct features.
    if 'sale_year' in X_train_eval.columns:
        if X_train_eval['sale_year'].isnull().any():
            # Use median from the current X_train_eval (which represents the full training set features)
            median_year_eval_calc = X_train_eval['sale_year'].median()
            median_year_eval = 0
            if pd.isna(median_year_eval_calc):
                print("Warning: All 'sale_year' values are missing in X_train_eval. Filling with 0.", file=sys.stderr)
            else:
                median_year_eval = int(median_year_eval_calc)

            print(
                f"Filling {X_train_eval['sale_year'].isnull().sum()} missing 'sale_year' values in X_train_eval with median: {median_year_eval}")
            X_train_eval['sale_year'] = X_train_eval['sale_year'].fillna(median_year_eval)

    # Ensure all feature columns are int/float (create_feature_engineered_data_from_list handles bool to int)
    # As a safeguard:
    for col in X_train_eval.columns:
        if X_train_eval[col].dtype == 'bool':
            X_train_eval[col] = X_train_eval[col].astype(int)
        # You might also want to check for object dtypes that should be numeric
        if X_train_eval[col].dtype == 'object':
            try:
                X_train_eval[col] = pd.to_numeric(X_train_eval[col])
                print(f"Warning: Column '{col}' was object type, converted to numeric.", file=sys.stderr)
            except ValueError:
                print(f"Error: Column '{col}' is object type and could not be converted to numeric.", file=sys.stderr)
                # Decide how to handle this, e.g., drop or raise error
                return

    # 5. Make Predictions on the training data
    try:
        y_pred_train = model.predict(X_train_eval)
    except Exception as e:
        print(f"Error during prediction on training data: {e}", file=sys.stderr)
        print(f"X_train_eval dtypes:\n{X_train_eval.dtypes}", file=sys.stderr)
        print(f"X_train_eval head:\n{X_train_eval.head()}", file=sys.stderr)
        return

    # 6. Evaluate Model on Training Set
    mse_train = mean_squared_error(y_train_actual, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train_actual, y_pred_train)

    print(f"Mean Squared Error (MSE): {mse_train:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse_train:.2f}")
    print(f"R-squared (R2 Score): {r2_train:.4f}")
    print("--------------------------------------")


# Removed: evaluate_local_model_on_ml_data function definition


def evaluate_per_record_local_curve_fitting():
    """
    Evaluates the local curve fitting model on a per-record basis using all ML data.
    For each unique record, it fits a curve and calculates metrics, then averages them.
    """
    print("\n--- Evaluating Local Curve Model (Per-Record Fit) on Full ML Data ---")

    raw_sales = read_ml_data()
    if not raw_sales:
        print("No raw sales data found for per-record local model evaluation.", file=sys.stderr)
        return

    # Group sales by record identifier
    # A record is defined by artist, album, label, and extra_comments
    records_data = defaultdict(lambda: {'qualities': [], 'prices': []})
    for sale in raw_sales:
        # Ensure all parts of the key are strings or a consistent type (e.g., None becomes '')
        record_key = (
            sale.get('artist', '') or '',
            sale.get('album', '') or '',
            sale.get('label', '') or '',
            sale.get('extra_comments', '') or ''
        )
        quality = sale.get('quality')
        price = sale.get('price')
        if quality is not None and price is not None:
            records_data[record_key]['qualities'].append(quality)
            records_data[record_key]['prices'].append(price)

    if not records_data:
        print("No records found to evaluate after grouping.", file=sys.stderr)
        return

    all_record_metrics = []
    evaluated_records_count = 0
    skipped_records_insufficient_data = 0

    for record_key, data in records_data.items():
        qualities_record = data['qualities']
        prices_record = data['prices']

        if len(qualities_record) < MIN_POINTS_PER_RECORD_FOR_FIT:
            # print(f"Skipping record {record_key}: Insufficient data points ({len(qualities_record)} < {MIN_POINTS_PER_RECORD_FOR_FIT}).", file=sys.stderr)
            skipped_records_insufficient_data += 1
            continue

        try:
            params_record, predict_func_record = fit_curve_and_get_params(qualities_record, prices_record)
            if params_record is None or predict_func_record is None:
                # print(f"Curve fitting failed for record: {record_key}", file=sys.stderr)
                continue

            # Predict prices for this record's actual quality scores
            predicted_prices_record = predict_func_record(np.array(qualities_record))

            # Calculate metrics for this record
            mse_record = mean_squared_error(prices_record, predicted_prices_record)
            rmse_record = np.sqrt(mse_record)
            r2_record = r2_score(prices_record, predicted_prices_record)

            all_record_metrics.append({
                'mse': mse_record,
                'rmse': rmse_record,
                'r2': r2_record
            })
            evaluated_records_count += 1

        except Exception as e:
            # print(f"Error processing record {record_key}: {e}", file=sys.stderr)
            continue  # Skip to next record on error

    if not all_record_metrics:
        print("No records were successfully evaluated for per-record local curve fitting.")
        if skipped_records_insufficient_data > 0:
            print(
                f"Skipped {skipped_records_insufficient_data} records due to insufficient data (less than {MIN_POINTS_PER_RECORD_FOR_FIT} points).")
        return

    # Calculate average metrics
    avg_mse = np.mean([m['mse'] for m in all_record_metrics])
    avg_rmse = np.mean([m['rmse'] for m in all_record_metrics])
    avg_r2 = np.mean([m['r2'] for m in all_record_metrics])  # Simple average of R2

    print(f"Evaluated {evaluated_records_count} unique records with sufficient data.")
    if skipped_records_insufficient_data > 0:
        print(
            f"Skipped {skipped_records_insufficient_data} records due to insufficient data (less than {MIN_POINTS_PER_RECORD_FOR_FIT} points).")
    print(f"Average Per-Record MSE: {avg_mse:.2f}")
    print(f"Average Per-Record RMSE: {avg_rmse:.2f}")
    print(f"Average Per-Record R2 Score: {avg_r2:.4f}")
    print("-----------------------------------------------------------------")


if __name__ == "__main__":
    train_and_evaluate_model()
