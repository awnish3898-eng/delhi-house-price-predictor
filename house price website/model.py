import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

DATASET_PATH = os.path.join(os.path.dirname(__file__), "cleaned_kaggle_housing.csv")
MODEL_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
HEATMAP_PATH = os.path.join(STATIC_DIR, "model_comparison_heatmap.png")

# Models dictionary cache
TRAINED_MODELS = {}
MODEL_METRICS = {
    "Random Forest": {"R2 Score": 0.808, "MAE (₹ Lakh)": 48.12, "RMSE (₹ Lakh)": 84.02},
    "Linear Regression": {"R2 Score": 0.793, "MAE (₹ Lakh)": 59.03, "RMSE (₹ Lakh)": 87.34},
    "Decision Tree": {"R2 Score": 0.751, "MAE (₹ Lakh)": 57.43, "RMSE (₹ Lakh)": 95.61}
}

def get_locations():
    """Get sorted list of available locations from cleaned dataset."""
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        return sorted(df["Location"].unique().tolist())
    return [
        "Alaknanda", "Chhattarpur", "Chittaranjan Park", "Dwarka", 
        "Greater Kailash", "Karol Bagh", "Lajpat Nagar", "Laxmi Nagar", 
        "Mehrauli", "New Friends Colony", "Paschim Vihar", "Patel Nagar", 
        "Rohini", "Safdarjung Enclave", "Saket", "Sarita Vihar", 
        "Sheikh Sarai", "Vasant Kunj", "Yamuna Vihar"
    ]

AVAILABLE_LOCATIONS = get_locations()

def get_metrics():
    """Return dictionary of model performance metrics."""
    global MODEL_METRICS
    return MODEL_METRICS

def format_inr(amount):
    """Format price in Indian Lakhs / Crores."""
    if amount >= 10000000:
        cr = amount / 10000000
        return f"₹ {cr:.2f} Crore"
    elif amount >= 100000:
        lakh = amount / 100000
        return f"₹ {lakh:.2f} Lakh"
    else:
        return f"₹ {amount:,.0f}"

def train_all_models():
    """Train Linear Regression, Decision Tree, and Random Forest models and generate comparison heatmap."""
    global TRAINED_MODELS, MODEL_METRICS
    
    os.makedirs(STATIC_DIR, exist_ok=True)
    df = pd.read_csv(DATASET_PATH)

    X = df[["Location", "Area", "Bedrooms", "Bathrooms"]]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Location"])
        ],
        remainder="passthrough"
    )

    regressors = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=8)
    }

    metrics_dict = {}

    for name, reg in regressors.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", reg)
        ])
        pipeline.fit(X_train, y_train)
        TRAINED_MODELS[name] = pipeline

        # Save individual model
        filename = f"{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(pipeline, os.path.join(MODEL_DIR, filename))

        # Evaluate on test set
        preds = pipeline.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        metrics_dict[name] = {
            "R2 Score": round(float(r2), 3),
            "MAE (₹ Lakh)": round(float(mae) / 100000, 2),
            "RMSE (₹ Lakh)": round(float(rmse) / 100000, 2)
        }

    MODEL_METRICS.update(metrics_dict)

    # Generate comparison heatmap
    generate_comparison_heatmap(df, metrics_dict)
    print("All models successfully trained and heatmap generated!")
    return TRAINED_MODELS, MODEL_METRICS

def generate_comparison_heatmap(df, metrics_dict):
    """Create a side-by-side heatmap showing Feature Correlations and Model Performance."""
    os.makedirs(STATIC_DIR, exist_ok=True)
    
    # 1. Feature Correlation Matrix
    num_cols = ["Area", "Bedrooms", "Bathrooms", "Price"]
    corr_matrix = df[num_cols].corr()

    # 2. Model Performance Matrix
    metrics_df = pd.DataFrame(metrics_dict).T

    # Setup matplotlib figure
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.patch.set_facecolor('#ffffff')

    # Subplot 1: Feature Correlation Heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        cbar=True, 
        ax=ax1, 
        linewidths=1.2, 
        linecolor="white",
        annot_kws={"size": 11, "weight": "bold"}
    )
    ax1.set_title("1. Feature Correlation Heatmap\n(Relationship with Price)", fontsize=13, fontweight="bold", pad=12)

    # Subplot 2: Model Performance Heatmap
    norm_df = metrics_df.copy()
    norm_df["R2 Score"] = (norm_df["R2 Score"] - norm_df["R2 Score"].min()) / (norm_df["R2 Score"].max() - norm_df["R2 Score"].min() + 1e-6)
    norm_df["MAE (₹ Lakh)"] = (norm_df["MAE (₹ Lakh)"].max() - norm_df["MAE (₹ Lakh)"]) / (norm_df["MAE (₹ Lakh)"].max() - norm_df["MAE (₹ Lakh)"].min() + 1e-6)
    norm_df["RMSE (₹ Lakh)"] = (norm_df["RMSE (₹ Lakh)"].max() - norm_df["RMSE (₹ Lakh)"]) / (norm_df["RMSE (₹ Lakh)"].max() - norm_df["RMSE (₹ Lakh)"].min() + 1e-6)

    sns.heatmap(
        norm_df, 
        annot=metrics_df.values, 
        fmt=".2f", 
        cmap="YlGn", 
        cbar=False, 
        ax=ax2, 
        linewidths=1.5, 
        linecolor="white",
        annot_kws={"size": 11, "weight": "bold"}
    )
    ax2.set_title("2. Model Evaluation Heatmap\n(R² Score, MAE & RMSE — Greener is Better)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_ylabel("Algorithm", fontweight="bold", fontsize=11)

    plt.tight_layout()
    plt.savefig(HEATMAP_PATH, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved at: {HEATMAP_PATH}")

def load_or_train():
    """Load models or train if not loaded yet."""
    global TRAINED_MODELS
    if not TRAINED_MODELS:
        train_all_models()
    return TRAINED_MODELS

def predict_property(location, area, bedrooms, bathrooms, selected_model="All"):
    """
    Predict property price using the chosen model or all models for comparison.
    """
    load_or_train()
    input_data = pd.DataFrame([{
        "Location": location,
        "Area": float(area),
        "Bedrooms": int(bedrooms),
        "Bathrooms": int(bathrooms)
    }])

    predictions = {}
    for name, model in TRAINED_MODELS.items():
        raw_pred = max(1200000, model.predict(input_data)[0])
        predictions[name] = {
            "raw_price": round(raw_pred),
            "formatted_price": format_inr(raw_pred)
        }

    return {
        "predictions": predictions,
        "selected_model": selected_model,
        "metrics": MODEL_METRICS
    }

if __name__ == "__main__":
    train_all_models()
    test_result = predict_property("Dwarka", 1200, 3, 2)
    print("\nPredictions for Dwarka (1200 sqft, 3 BHK, 2 Bath):")
    for m, p in test_result["predictions"].items():
        print(f"  - {m:18}: {p['formatted_price']}")
