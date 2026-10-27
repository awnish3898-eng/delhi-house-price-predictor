import os
from app import app
from model import predict_property, AVAILABLE_LOCATIONS, get_metrics

def run_tests():
    print("Testing multi-model predictions:")
    res = predict_property("Dwarka", 1200, 3, 2)
    assert "Linear Regression" in res["predictions"], "Linear Regression missing from predictions"
    assert "Decision Tree" in res["predictions"], "Decision Tree missing from predictions"
    assert "Random Forest" not in res["predictions"], "Random Forest should be removed from predictions"

    for model_name, pred in res["predictions"].items():
        print(f"  {model_name:18} -> {pred['formatted_price']}")

    print("\nVerifying model metrics:")
    metrics = get_metrics()
    assert "Linear Regression" in metrics, "Linear Regression missing from metrics"
    assert "Decision Tree" in metrics, "Decision Tree missing from metrics"
    assert "Random Forest" not in metrics, "Random Forest should be removed from metrics"

    for model_name, met in metrics.items():
        print(f"  {model_name:18} -> R2: {met['R2 Score']}, MAE: {met['MAE (₹ Lakh)']}L")

    print("\nTesting Multipage Flask Endpoints:")
    with app.test_client() as client:
        # Page 1: GET / (Calculator)
        res_home = client.get("/")
        assert res_home.status_code == 200
        assert b"Valuation Calculator" in res_home.data
        assert b"Dwarka" in res_home.data
        assert b"Random Forest" not in res_home.data, "Random Forest should not appear on home page"
        print("  [PASS] Page 1: GET / (Home / Calculator) loaded")

        # Test pre-selection via query parameter (?location=Rohini)
        res_query = client.get("/?location=Rohini")
        assert res_query.status_code == 200
        assert b'value="Rohini" selected' in res_query.data
        print("  [PASS] Page 1: Query param pre-selection (?location=Rohini) works")

        # Page 2: GET /map (Territory Map)
        res_map = client.get("/map")
        assert res_map.status_code == 200
        assert b"Interactive Territory Map" in res_map.data
        assert b"fullMap" in res_map.data
        print("  [PASS] Page 2: GET /map (Interactive Map & Territory Validator) loaded")

        # Verify /analytics is removed and returns 404
        res_analytics = client.get("/analytics")
        assert res_analytics.status_code == 404
        print("  [PASS] /analytics route is removed and returns 404")

        # Page 4: GET /about (Project Documentation & Viva Guide)
        res_about = client.get("/about")
        assert res_about.status_code == 200
        assert b"Project Documentation & Architecture" in res_about.data
        assert b"clean_kaggle_data.py" in res_about.data
        assert b"Random Forest" not in res_about.data, "Random Forest should not appear in About documentation"
        print("  [PASS] Page 4: GET /about (Project Documentation & Viva Guide) loaded")

        # POST / with valid location (Dwarka) and All models
        valid_res = client.post("/", data={
            "location": "Dwarka",
            "area": "1500",
            "bedrooms": "3",
            "bathrooms": "2",
            "model_choice": "All (Compare All Models)"
        })
        assert valid_res.status_code == 200
        assert b"Linear Regression" in valid_res.data
        assert b"Decision Tree" in valid_res.data
        assert b"Random Forest" not in valid_res.data, "Random Forest should not appear in results"
        print("  [PASS] POST / (All Models): Generates valuation result cards without Random Forest")

        # POST / with Decision Tree specifically selected
        dt_res = client.post("/", data={
            "location": "Dwarka",
            "area": "1500",
            "bedrooms": "3",
            "bathrooms": "2",
            "model_choice": "Decision Tree"
        })
        assert dt_res.status_code == 200
        assert b"Decision Tree" in dt_res.data
        print("  [PASS] POST / (Decision Tree): Predicts successfully")

        # POST / with Linear Regression specifically selected
        lr_res = client.post("/", data={
            "location": "Dwarka",
            "area": "1500",
            "bedrooms": "3",
            "bathrooms": "2",
            "model_choice": "Linear Regression"
        })
        assert lr_res.status_code == 200
        assert b"Linear Regression" in lr_res.data
        print("  [PASS] POST / (Linear Regression): Predicts successfully")

        # POST / with invalid location
        invalid_res = client.post("/", data={
            "location": "Jaipur",
            "area": "1500",
            "bedrooms": "3",
            "bathrooms": "2",
            "model_choice": "All (Compare All Models)"
        })
        assert invalid_res.status_code == 200
        assert b"We do not operate in" in invalid_res.data
        print("  [PASS] POST /: Rejects unsupported city (Jaipur) cleanly")

    print("\nALL MULTIPAGE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
