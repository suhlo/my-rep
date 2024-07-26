# Save this as app.py
from flask import Flask, request, jsonify, render_template
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Ensure the results are consistent by setting seed
DetectorFactory.seed = 0

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        text = request.form["text"]
        try:
            language = detect(text)
            return jsonify({"language": language})
        except LangDetectException:
            return jsonify({"language": "Could not detect language"})


if __name__ == "__main__":
    app.run(debug=True)
