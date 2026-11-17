"""
Rust Detection MVP - Training Script
Objective: Train a Random Forest Regressor to predict rust percentage
from electric field measurements (Real and Imaginary components).
"""

import os
import re
import glob
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def load_and_combine_data(dataset_dir="dataset", sample_size_per_file=None):
    """
    Reads all Excel files from dataset folder, verifies compatibility,
    extracts Real and Imaginary electric field measurements along with Rust %,
    and combines them into a single DataFrame while preserving sample identity.
    """
    print(f"Scanning for Excel files in '{dataset_dir}'...")
    excel_files = sorted(glob.glob(os.path.join(dataset_dir, "*.xlsx")))
    
    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in '{dataset_dir}'.")
    
    print(f"Found {len(excel_files)} Excel files.")
    
    data_frames = []
    incompatible_files = []
    
    for file_path in excel_files:
        file_name = os.path.basename(file_path)
        
        # Extract sample identity and component from filename (e.g., Ex_Field_1.xlsx)
        match = re.search(r'(E[xy])_Field_(\d+)', file_name, re.IGNORECASE)
        component = match.group(1).upper() if match else "E"
        sample_id = int(match.group(2)) if match else 1
        
        try:
            raw_df = pd.read_excel(file_path, header=None)
            
            # Basic compatibility checks
            if raw_df.shape[0] < 3 or raw_df.shape[1] < 3:
                incompatible_files.append((file_name, f"Insufficient rows/cols: {raw_df.shape}"))
                continue
                
            first_cell = str(raw_df.iloc[0, 0]).strip()
            re_label = str(raw_df.iloc[1, 1]).strip()
            im_label = str(raw_df.iloc[1, 2]).strip()
            
            if "Rust" not in first_cell or re_label != "Re" or im_label != "Im":
                incompatible_files.append(
                    (file_name, f"Unexpected header format: '{first_cell}', '{re_label}', '{im_label}'")
                )
                continue
            
            # Row 0: Rust percentages (starting column 1, step 2)
            rust_pcts = pd.to_numeric(raw_df.iloc[0, 1::2].values, errors='coerce')
            
            # Rows 2+: Real and Imaginary field measurements
            re_matrix = raw_df.iloc[2:, 1::2].values.astype(float)
            im_matrix = raw_df.iloc[2:, 2::2].values.astype(float)
            
            n_points, n_rust = re_matrix.shape
            
            # Flatten 2D matrices into 1D columns
            re_flat = re_matrix.flatten()
            im_flat = im_matrix.flatten()
            rust_flat = np.tile(rust_pcts, n_points)
            points_flat = np.repeat(np.arange(n_points), n_rust)
            
            file_df = pd.DataFrame({
                "real_electric_field": re_flat,
                "imaginary_electric_field": im_flat,
                "rust_percentage": rust_flat,
                "sample_id": sample_id,
                "component": component,
                "measurement_point": points_flat,
                "source_file": file_name
            })
            
            # Handle missing values if any
            initial_count = len(file_df)
            file_df = file_df.dropna(subset=["real_electric_field", "imaginary_electric_field", "rust_percentage"])
            dropped = initial_count - len(file_df)
            if dropped > 0:
                print(f"  [{file_name}] Handled missing values: dropped {dropped} rows.")
                
            data_frames.append(file_df)
            print(f"  [Loaded] {file_name} -> {len(file_df):,} measurements (Sample {sample_id}, {component})")
            
        except Exception as e:
            incompatible_files.append((file_name, str(e)))
            
    if incompatible_files:
        print("\nWARNING: Some files were incompatible:")
        for fname, reason in incompatible_files:
            print(f"  - {fname}: {reason}")
            
    if not data_frames:
        raise ValueError("No valid data could be extracted from Excel files.")
        
    combined_df = pd.concat(data_frames, ignore_index=True)
    print(f"\nSuccessfully combined {len(data_frames)} files.")
    print(f"Total dataset shape: {combined_df.shape[0]:,} rows, {combined_df.shape[1]} columns.")
    
    return combined_df


def train_model(df, models_dir="models", max_samples=250000):
    """
    Trains a Random Forest Regressor on electric field measurements
    with an 80:20 train-test split that preserves sample identity to prevent data leakage.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Subsample if dataset is very large to ensure fast training within the MVP timeframe
    total_rows = len(df)
    if total_rows > max_samples:
        print(f"\nSubsampling {max_samples:,} representative records out of {total_rows:,} for fast MVP training...")
        samples_list = []
        samples_per_id = max_samples // df['sample_id'].nunique()
        for sid in sorted(df['sample_id'].unique()):
            sid_df = df[df['sample_id'] == sid]
            n_take = min(len(sid_df), samples_per_id)
            samples_list.append(sid_df.sample(n=n_take, random_state=42))
        df_train_ready = pd.concat(samples_list, ignore_index=True)
    else:
        df_train_ready = df
        
    features = ["real_electric_field", "imaginary_electric_field"]
    target = "rust_percentage"
    
    X = df_train_ready[features]
    y = df_train_ready[target]
    groups = df_train_ready["sample_id"]
    
    print("\nSplitting dataset into 80:20 train-test split (preserving sample identity)...")
    unique_samples = df_train_ready["sample_id"].nunique()
    
    if unique_samples >= 5:
        # GroupShuffleSplit ensures test samples are never seen in training (avoids data leakage)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        train_samples = groups.iloc[train_idx].unique().tolist()
        test_samples = groups.iloc[test_idx].unique().tolist()
        print(f"  Train samples (IDs): {train_samples}")
        print(f"  Test samples (IDs): {test_samples}")
    else:
        # Fallback to standard 80:20 split if fewer samples
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    print(f"  Train set: {len(X_train):,} records ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test set:  {len(X_test):,} records ({len(X_test)/len(X)*100:.1f}%)")
    
    print("\nTraining Random Forest Regressor...")
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=16,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.2f} seconds.")
    
    print("\nEvaluating model on test set...")
    y_pred = rf.predict(X_test)
    
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(root_mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    
    print("=" * 45)
    print("           MODEL EVALUATION RESULTS          ")
    print("=" * 45)
    print(f"  Mean Absolute Error (MAE) : {mae:.4f} %")
    print(f"  Root Mean Squared Error   : {rmse:.4f} %")
    print(f"  R-squared (R2 Score)      : {r2:.4f}")
    print("=" * 45)
    
    # Save the model
    model_path = os.path.join(models_dir, "rust_model.joblib")
    joblib.dump(rf, model_path)
    print(f"\nTrained model successfully saved to: {model_path}")
    
    # Save metrics metadata for Streamlit app
    metrics_info = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
        "train_records": len(X_train),
        "test_records": len(X_test),
        "total_records": total_rows,
        "n_samples": unique_samples,
        "features": features,
        "target": target,
        "model_type": "Random Forest Regressor",
        "training_time_sec": round(train_time, 2)
    }
    
    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_info, f, indent=4)
    print(f"Metrics metadata saved to: {metrics_path}")
    
    return rf, metrics_info


if __name__ == "__main__":
    combined_data = load_and_combine_data()
    train_model(combined_data)
