import os
import io
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ─── Config ───────────────────────────────────────────────────────────────────
USGS_URL = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query?"
    "format=csv&starttime=2022-01-01&endtime=2022-12-31"
    "&minmagnitude=2.5&limit=10000&orderby=time"
)
RAW_FILE = "earthquake_raw.csv"
OUTPUT_DIR = "earthquake_preprocessing"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "earthquake_preprocessing.csv")

FEATURE_COLS = [
    "latitude", "longitude", "depth", "magType",
    "nst", "gap", "dmin", "rms",
    "horizontalError", "depthError", "magError", "magNst",
]
TARGET_COL = "significant"
MAG_THRESHOLD = 5.0

NUMERIC_COLS = [
    "latitude", "longitude", "depth",
    "nst", "gap", "dmin", "rms",
    "horizontalError", "depthError", "magError", "magNst",
]


# ─── Step 1: Download ─────────────────────────────────────────────────────────
def download_earthquake_data(output_path: str = RAW_FILE) -> pd.DataFrame:
    """Download earthquake data from the USGS public API."""
    print("[1/5] Downloading earthquake data from USGS …")
    response = requests.get(USGS_URL, timeout=120)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    df.to_csv(output_path, index=False)
    print(f"      Saved {len(df):,} records → {output_path}")
    return df


def load_data(filepath: str = RAW_FILE) -> pd.DataFrame:
    """Load raw earthquake CSV."""
    return pd.read_csv(filepath)


# ─── Step 2: Create target ────────────────────────────────────────────────────
def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary target column: 1 = significant (mag ≥ 5.0), 0 = otherwise."""
    print("[2/5] Creating target variable …")
    df = df.copy()
    df[TARGET_COL] = (df["mag"] >= MAG_THRESHOLD).astype(int)
    dist = df[TARGET_COL].value_counts(normalize=True).mul(100).round(1)
    print(f"      Class distribution — 0: {dist.get(0, 0)}%  1: {dist.get(1, 0)}%")
    return df


# ─── Step 3: Select & clean features ─────────────────────────────────────────
def select_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Select feature columns + target; impute missing values."""
    print("[3/5] Selecting features and imputing missing values …")
    cols = FEATURE_COLS + [TARGET_COL]
    df = df[cols].copy()

    before = df.isnull().sum().sum()
    for col in NUMERIC_COLS:
        df[col] = df[col].fillna(df[col].median())

    df["magType"] = df["magType"].fillna("unknown").astype(str).str.strip().str.lower()
    after = df.isnull().sum().sum()
    print(f"      Missing values: {before:,} → {after:,}")
    return df


# ─── Step 4: Encode & scale ───────────────────────────────────────────────────
def encode_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode magType, then StandardScale all numeric features."""
    print("[4/5] Encoding and scaling …")

    le = LabelEncoder()
    df["magType"] = le.fit_transform(df["magType"])

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLS, index=X.index)

    result = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
    print(f"      Final shape: {result.shape}")
    return result


# ─── Step 5: Save ─────────────────────────────────────────────────────────────
def save_data(df: pd.DataFrame, output_path: str = OUTPUT_FILE) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[5/5] Preprocessed data saved → {output_path}")


# ─── Pipeline ─────────────────────────────────────────────────────────────────
def run_pipeline(raw_path: str = RAW_FILE) -> pd.DataFrame:
    if not os.path.exists(raw_path):
        download_earthquake_data(raw_path)

    df = load_data(raw_path)
    df = create_target(df)
    df = select_and_clean(df)
    df = encode_and_scale(df)
    save_data(df)
    return df


if __name__ == "__main__":
    result = run_pipeline()
    print("\nDone! Sample:")
    print(result.head())
