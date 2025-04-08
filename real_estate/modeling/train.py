import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("real_estate/logs/train.log"),
        logging.StreamHandler(sys.stdout),
    ],
)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        logging.error(f"File not found: {path}")
        raise FileNotFoundError(f"Data file not found at {path}")
    logging.info(f"Loading data from {path}")
    return pd.read_csv(path)


def split_features_target(
    df: pd.DataFrame, target_col: str = "price"
) -> tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features and target."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    stratify_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split data into train/test sets using stratified sampling.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        stratify_col: Column name in X to stratify by.
        test_size: Fraction of data for test set.
        random_state: Random seed for reproducibility.
    """
    if stratify_col not in X.columns:
        raise ValueError(f"'{stratify_col}' not found in features.")
    return train_test_split(
        X, y, test_size=test_size, stratify=X[stratify_col], random_state=random_state
    )


def train_linear_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Train a linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: LinearRegression, X: pd.DataFrame, y: pd.Series, dataset_name: str = ""
):
    """Evaluate model performance using MAE and R²."""
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"{dataset_name} MAE: {mae:.2f}")
    print(f"{dataset_name} R²: {r2:.2f}")
    return preds, mae, r2


def main(data_path: str = "data/raw/final.csv") -> LinearRegression:
    """Main function to run data loading, training, and evaluation."""
    try:
        df = load_data(data_path)
        X, y = split_features_target(df)

        X_train, X_test, y_train, y_test = stratified_split(
            X, y, stratify_col="property_type_Bunglow"
        )

        model = train_linear_model(X_train, y_train)

        print("\n--- Training Evaluation ---")
        evaluate_model(model, X_train, y_train, "Train")

        print("\n--- Testing Evaluation ---")
        evaluate_model(model, X_test, y_test, "Test")

        return model

    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
