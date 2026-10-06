# 🏡 Basic Noida House Price Predictor

A simple Machine Learning web application built using **Python, Flask, and scikit-learn** to predict residential house prices across key areas in **Noida (Uttar Pradesh)**.

---

## 📌 Features
- **Noida-Specific Real Estate Data**: Reflects realistic market prices for prominent Noida locations:
  - Noida Extension (Greater Noida West)
  - Sector 62 (Central / IT Hub)
  - Sector 137 (Noida-Greater Noida Expressway)
  - Sector 50 (Prime Residential Sector)
  - Sector 150 (Expressway Green & Luxury Corridor)
- **Inputs**:
  - Location / Sector
  - Area (Square Feet)
  - Bedrooms (BHK)
  - Property Age (Years)
- **Output**: Price formatted in Indian currency (**₹ Lakhs** and **₹ Crores**).
- **Clean UI**: Clean, mobile-friendly design that runs directly in your browser.

---

## 🚀 How to Run the Website

### Step 1: Open Terminal / Command Prompt
Navigate to this folder:
```bash
cd "C:\Users\HP\Desktop\house price website"
```

### Step 2: Start the Flask App
Run:
```bash
python app.py
```

### Step 3: Open in Browser
Open your browser and visit:
```
http://127.0.0.1:5000
```

---

## 🧪 How to Run Tests
To verify all predictions and routes:
```bash
python test_app.py
```

---

## 📁 Project Structure
- `app.py`: Flask web server and form handling.
- `model.py`: Linear Regression training and prediction logic.
- `noida_housing.csv`: Dataset with representative Noida housing rates.
- `noida_model.pkl`: Saved pre-trained model.
- `templates/index.html`: Clean webpage interface.
- `test_app.py`: Automated tests to check accuracy and routes.
