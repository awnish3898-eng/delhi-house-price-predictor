import os
from flask import Flask, render_template, request
from model import predict_property, AVAILABLE_LOCATIONS, get_metrics, load_or_train

app = Flask(__name__)

# Preload models on startup
load_or_train()

MODEL_OPTIONS = [
    "All (Compare All Models)",
    "Random Forest",
    "Linear Regression",
    "Decision Tree"
]

@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    error_message = None
    
    # Check if a location was passed via query parameter from the map page
    preselected_loc = request.args.get("location", "").strip()
    if preselected_loc and preselected_loc in AVAILABLE_LOCATIONS:
        default_location = preselected_loc
    else:
        default_location = AVAILABLE_LOCATIONS[0] if AVAILABLE_LOCATIONS else "Dwarka"

    form_data = {
        "location": default_location,
        "area": "1200",
        "bedrooms": "2",
        "bathrooms": "2",
        "model_choice": "All (Compare All Models)"
    }

    if request.method == "POST":
        try:
            location = request.form.get("location", "").strip()
            area = request.form.get("area", "").strip()
            bedrooms = request.form.get("bedrooms", "2").strip()
            bathrooms = request.form.get("bathrooms", "2").strip()
            model_choice = request.form.get("model_choice", "All (Compare All Models)")

            form_data = {
                "location": location,
                "area": area,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "model_choice": model_choice
            }

            # Check if location is supported
            if not location:
                error_message = "Please select or click a valid operating location."
            elif location not in AVAILABLE_LOCATIONS:
                error_message = f"❌ Service Unavailable: We do not operate in '{location}'. Please select an active Delhi zone."
            elif not area or not bedrooms or not bathrooms:
                error_message = "Please fill in all the required fields."
            else:
                area_val = float(area)
                bedrooms_val = int(bedrooms)
                bathrooms_val = int(bathrooms)

                if area_val <= 0 or bedrooms_val <= 0 or bathrooms_val <= 0:
                    error_message = "Please enter positive numbers only."
                elif area_val < 200 or area_val > 15000:
                    error_message = "Area must be between 200 and 15,000 sq ft."
                else:
                    results = predict_property(
                        location, 
                        area_val, 
                        bedrooms_val, 
                        bathrooms_val,
                        selected_model=model_choice
                    )

        except ValueError:
            error_message = "Invalid input! Please enter numeric values."
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"

    return render_template(
        "index.html",
        active_page="home",
        locations=AVAILABLE_LOCATIONS,
        model_options=MODEL_OPTIONS,
        results=results,
        metrics=get_metrics(),
        error_message=error_message,
        form_data=form_data
    )

@app.route("/map")
def map_page():
    return render_template(
        "map.html",
        active_page="map",
        locations=AVAILABLE_LOCATIONS
    )

@app.route("/about")
def about_page():
    return render_template(
        "about.html",
        active_page="about"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Multipage EstateAI Platform on port {port} ...")
    app.run(host="0.0.0.0", port=port, debug=False)