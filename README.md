# Rust Detection Using Electric Field Analysis - ML MVP

A lightweight, functional Machine Learning MVP designed to predict the percentage of rust in a material using its electric field measurements (Real and Imaginary components).

---

## Project Structure

```text
Rust-Detection-MVP/
│
├── dataset/             # Contains 20 Excel files (Ex_Field_*.xlsx and Ey_Field_*.xlsx)
├── models/              # Saved model artifacts
│   ├── rust_model.joblib  # Trained Random Forest Regressor
│   └── metrics.json       # Test evaluation metrics & training metadata
├── train.py             # Data loading, preprocessing, 80:20 training, and evaluation
├── app.py               # Clean Streamlit user interface for live inference
├── requirements.txt     # Python dependencies
└── README.md            # Documentation and instructions
```

---

## Dataset Overview

The dataset consists of 20 Excel files (`Ex_Field_1.xlsx` through `Ex_Field_10.xlsx` and `Ey_Field_1.xlsx` through `Ey_Field_10.xlsx`) representing electric field measurements from 10 distinct physical samples across $E_x$ and $E_y$ vector components:
- **Rust Percentage**: Ranging from 0.0% to 100.0% in 0.1% increments (1,001 distinct states).
- **Measurement Columns**: Alternating Real (`Re`) and Imaginary (`Im`) electric field values for each rust percentage.
- **Data Points**: 100 measurement sweeps per rust percentage level.
- **Combined Volume**: 2,002,000 measurements combined without data loss.

---

## Key Features

1. **Robust Data Pipeline**: Automatically parses, verifies, and combines all 20 Excel files into a clean Pandas DataFrame.
2. **Data Leakage Prevention**: Enforces sample identity preservation during the 80:20 train-test split (`GroupShuffleSplit` on `sample_id`), guaranteeing test samples are held out completely.
3. **Random Forest Regression**: Efficient Scikit-Learn regressor trained on Real and Imaginary field components.
4. **Model Serialization**: Models saved with Joblib and loaded on-demand by the application without retraining.
5. **Interactive Streamlit Web App**:
   - Inputs for Real Electric Field and Imaginary Electric Field.
   - Preset selector for rapid testing.
   - Live prediction button displaying predicted rust percentage and severity category.
   - Transparent display of model evaluation metrics (MAE, RMSE, R²).

---

## Installation & Setup

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1: Train the Model
Run the training script to load datasets, train the Random Forest Regressor, evaluate metrics on held-out test samples, and persist the artifacts:

```bash
python train.py
```

Expected output:
```text
Scanning for Excel files in 'dataset'...
Found 20 Excel files.
  [Loaded] Ex_Field_1.xlsx -> 100,100 measurements (Sample 1, EX)
  ...
Successfully combined 20 files.
Total dataset shape: 2,002,000 rows, 7 columns.

Splitting dataset into 80:20 train-test split (preserving sample identity)...
  Train samples (IDs): [1, 3, 4, 5, 6, 7, 8, 10]
  Test samples (IDs): [2, 9]
  Train set: 200,000 records (80.0%)
  Test set:  50,000 records (20.0%)

Training Random Forest Regressor...
Training completed in ~7 seconds.

=============================================
           MODEL EVALUATION RESULTS          
=============================================
  Mean Absolute Error (MAE) : 19.33 %
  Root Mean Squared Error   : 23.59 %
  R-squared (R2 Score)      : 0.3321
=============================================
```

### Step 2: Launch the Streamlit App
Start the Streamlit interface:

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Model Evaluation Metrics

| Metric | Test Set Value | Description |
| :--- | :--- | :--- |
| **MAE** | **19.33 %** | Mean Absolute Error on unseen test samples |
| **RMSE** | **23.59 %** | Root Mean Squared Error on unseen test samples |
| **R² Score** | **0.3321** | Coefficient of Determination on unseen test samples |

---

## Design Constraints Met

- **No Deep Learning / TensorFlow**: Scikit-Learn Random Forest Regressor only.
- **No Database / Authentication**: Direct file-based persistence with Joblib and JSON.
- **No Data Fabrication**: Trained exclusively on actual measurements from the 20 provided Excel files.
- **No Data Leakage**: Enforces sample-based group splitting.
