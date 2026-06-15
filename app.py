from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import pickle
import numpy as np

app = Flask(__name__)

model = load_model("model_ann.h5", compile=False)
scaler = pickle.load(open("min_max_scaler.pkl", "rb"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/house", methods=["POST"])
def house():

    longitude = request.form["longitude"]
    latitude = request.form["latitude"]
    houseage = request.form["houseage"]
    houserooms = request.form["houserooms"]
    totlabedrooms = request.form["totlabedrooms"]
    population = request.form["population"]
    households = request.form["households"]
    medianincome = request.form["medianincome"]
    oceanproximity = request.form["oceanproximity"]

    rooms_per_household = request.form["rooms_per_household"]
    bedrooms_per_room = request.form["bedrooms_per_room"]
    population_per_household = request.form["population_per_household"]

    features = np.array([
        longitude,
        latitude,
        houseage,
        houserooms,
        totlabedrooms,
        population,
        households,
        medianincome,
        oceanproximity,
        rooms_per_household,
        bedrooms_per_room,
        population_per_household
    ], dtype=float)

    features_scaled = scaler.transform([features])

    prediction = model.predict(features_scaled)

    price = round(float(prediction[0][0]), 2)

    return render_template("index.html", result=price)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/doc")
def doc():
    return render_template("doc.html")


@app.route("/ann")
def ann():
    return render_template("ann.html")


if __name__ == "__main__":
    app.run(debug=True)