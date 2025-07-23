from src.data_loader import extract_and_load
from src.preprocessing import preprocess
from src.xgb_model import train_xgb
import numpy as np
import pandas as pd

def train_model(zip_file, extract_dir):
    # Step 1: Load dataset
    df = extract_and_load(zip_file, extract_dir)
    print("Dataset loaded:", df.shape)

    # Step 2: Preprocess
    X_train, X_test, y_train, y_test, encoder = preprocess(df)
    print("Preprocessing done. Train shape:", X_train.shape)

    # Show first 5 rows of X_train in clear format (like your screenshot)
    np.set_printoptions(suppress=True, precision=8)
    print("X_train preview:\n", pd.DataFrame(X_train[:5]).to_numpy())

    # Step 3: Train XGBoost model
    model, y_pred_log = train_xgb(X_train, y_train, X_test, y_test)
    print("Model training completed.")

    # Step 4: Reverse log transformation
    y_test_actual = np.expm1(y_test)
    y_pred_actual = np.expm1(y_pred_log)

    return y_test_actual, y_pred_actual
