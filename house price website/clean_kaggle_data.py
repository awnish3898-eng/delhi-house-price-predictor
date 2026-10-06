"""
Script: clean_kaggle_data.py
Description: Cleans raw Kaggle dataset (kaggle_magicbricks.csv),
extracts clean recognized localities, handles missing values, removes outliers,
and outputs cleaned_kaggle_housing.csv.
"""

import sys
import pandas as pd
import numpy as np

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# List of major recognized localities in Delhi NCR
PROMINENT_LOCALITIES = [
    "Dwarka",
    "Rohini",
    "Lajpat Nagar",
    "Greater Kailash",
    "Vasant Kunj",
    "Saket",
    "Karol Bagh",
    "Patel Nagar",
    "Laxmi Nagar",
    "Paschim Vihar",
    "Alaknanda",
    "Safdarjung Enclave",
    "New Friends Colony",
    "Chhattarpur",
    "Mehrauli",
    "Yamuna Vihar",
    "Sarita Vihar",
    "Chittaranjan Park",
    "Sheikh Sarai"
]

def extract_clean_locality(text):
    if not isinstance(text, str):
        return None
    text_lower = text.lower()
    for loc in PROMINENT_LOCALITIES:
        if loc.lower() in text_lower:
            return loc
    return None

def clean_data():
    print("Loading kaggle_magicbricks.csv...")
    df = pd.read_csv("kaggle_magicbricks.csv")

    # 1. Clean and map locality
    df["Location"] = df["Locality"].apply(extract_clean_locality)
    df = df.dropna(subset=["Location"]).copy()

    # 2. Impute Bathrooms if missing (use BHK)
    df["Bathroom"] = df["Bathroom"].fillna(df["BHK"])

    # 3. Filter outliers
    df_clean = df[
        (df["Area"] >= 350) & (df["Area"] <= 7000) &
        (df["Price"] >= 1500000) & (df["Price"] <= 120000000) &
        (df["BHK"] >= 1) & (df["BHK"] <= 5) &
        (df["Bathroom"] >= 1) & (df["Bathroom"] <= 6)
    ].copy()

    # 4. Standard clean dataframe
    cleaned_df = pd.DataFrame({
        "Location": df_clean["Location"],
        "Area": df_clean["Area"].round(0).astype(int),
        "Bedrooms": df_clean["BHK"].astype(int),
        "Bathrooms": df_clean["Bathroom"].astype(int),
        "Price": df_clean["Price"].astype(int)
    })

    # Sort alphabetically by location
    cleaned_df = cleaned_df.sort_values(by="Location").reset_index(drop=True)

    output_path = "cleaned_kaggle_housing.csv"
    cleaned_df.to_csv(output_path, index=False)
    print(f"Cleaned dataset successfully created at '{output_path}' with {len(cleaned_df)} rows!")
    print("\nLocality distribution in cleaned dataset:")
    print(cleaned_df["Location"].value_counts())
    return cleaned_df

if __name__ == "__main__":
    clean_data()
