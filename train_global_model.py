import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import numpy as np  # For np.sqrt
import sys  # For printing to stderr

# Assuming machine_learning.py is in the same directory or accessible in PYTHONPATH
from machine_learning import generate_features_for_ml_training

MODEL_FILENAME = "global_price_model.pkl"
FEATURES_FILENAME = "model_features.pkl"

# sets the random state - set to None for truly random, or 42 to be consistent.
random_state = 42

def train_and_evaluate_model():
    """
    Loads feature-engineered data, trains a RandomForestRegressor model,
    evaluates it, and saves the model and its feature list.
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

    # 3. Prepare X (features) and y (target)
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
        # Drop rows where target is NaN from both X and y
        valid_indices = y.dropna().index
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]
        if X.empty:
            print("Error: No valid data remaining after dropping rows with missing target values.", file=sys.stderr)
            return
        print(f"Shape after dropping rows with missing target: X={X.shape}, y={y.shape}")

    # 4. Handle Missing Values in X (specifically for 'sale_year')
    if 'sale_year' in X.columns:
        if X['sale_year'].isnull().any():
            median_year = X['sale_year'].median()
            if pd.isna(median_year):  # If all are NaN, median will be NaN
                # Fallback if median cannot be calculated (e.g., all years are None)
                # Choose a sensible default or raise an error.
                # For now, let's use a placeholder like 0, but this might need refinement.
                print("Warning: All 'sale_year' values are missing. Filling with 0.", file=sys.stderr)
                median_year = 0
            else:
                median_year = int(median_year)  # Ensure it's an integer
            print(f"Filling {X['sale_year'].isnull().sum()} missing 'sale_year' values with median: {median_year}")
            X['sale_year'] = X['sale_year'].fillna(median_year)
    else:
        print("Warning: 'sale_year' column not found in features.", file=sys.stderr)

    # Convert boolean columns to int (0 or 1) explicitly if they aren't already
    # Scikit-learn usually handles True/False correctly, but this is safer.
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)

    # --- Ensure all feature columns are numeric ---
    # This is important as RandomForestRegressor expects numeric input.
    # One-hot encoded features (True/False) become 1/0.
    # 'quality_score' and 'sale_year' are numeric.
    # If other non-numeric columns were accidentally included, this would be an issue.
    non_numeric_cols = X.select_dtypes(exclude=np.number).columns
    if len(non_numeric_cols) > 0:
        print(f"Error: Non-numeric columns found in features after pre-processing: {list(non_numeric_cols)}",
              file=sys.stderr)
        print("Please ensure all features are numeric or properly encoded.", file=sys.stderr)
        return

    if X.empty:
        print("Error: Feature set X is empty. Cannot proceed with training.", file=sys.stderr)
        return

    print(f"Final feature set shape for training: {X.shape}")

    # 5. Split Data
    # test_size=0.2 means 20% of data is for testing, 80% for training.
    # random_state ensures reproducibility of the split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    print(f"Data split into training and testing sets: X_train: {X_train.shape}, X_test: {X_test.shape}")

    # 6. Initialize Model
    # n_estimators: number of trees in the forest.
    # random_state: for reproducibility of the model training.
    # You can tune other hyperparameters later (e.g., max_depth, min_samples_split).
    model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True, n_jobs=-1)
    print("RandomForestRegressor model initialized.")

    # 7. Train Model
    print("Training the model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    if hasattr(model, 'oob_score_') and model.oob_score_:
        print(f"Model Out-of-Bag (OOB) R^2 score: {model.oob_score_:.4f}")

    # 8. Make Predictions on Test Set
    y_pred = model.predict(X_test)

    # 9. Evaluate Model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation on Test Set ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2 Score): {r2:.4f}")
    print("------------------------------------")

    # 10. Save the Model and Feature List
    try:
        with open(MODEL_FILENAME, 'wb') as f:
            pickle.dump(model, f)
        print(f"Trained model saved to {MODEL_FILENAME}")

        # Save the feature names (columns) in the order they were used for training
        feature_names = list(X_train.columns)
        with open(FEATURES_FILENAME, 'wb') as f:
            pickle.dump(feature_names, f)
        print(f"Model feature list saved to {FEATURES_FILENAME}")

    except Exception as e:
        print(f"Error saving model or features: {e}", file=sys.stderr)


if __name__ == "__main__":
    train_and_evaluate_model()
