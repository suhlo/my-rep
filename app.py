from flask import Flask, request, jsonify, render_template
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Ensure the results are consistent by setting seed
DetectorFactory.seed = 0

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the dataset (assuming it's a CSV file)
df = pd.read_csv("C:/Users/Suhas sattigeri/Desktop/Mini P/data/dataset1.csv")

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the dataset and transform the text data
X = df["Text"]
y = df["Language"]
X_tfidf = vectorizer.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Transform the training and test sets using the TF-IDF vectorizer
X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Train a Decision Tree classifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_tfidf, y_train)


@app.route("/")
def main():
    return render_template("main.html")


@app.route("/classify")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        text = request.form.get("text")
        if not text:
            return jsonify({"error": "Missing or empty text field"}), 400
        try:
            language = detect(text)
            # Transform the input text using the same TF-IDF vectorizer
            text_tfidf = vectorizer.transform([text])
            # Predict using Naive Bayes and Decision Tree models
            nb_prediction = nb_model.predict(text_tfidf)[0]
            dt_prediction = dt_model.predict(text_tfidf)[0]
            return jsonify(
                {
                    "language": language,
                    "nb_prediction": nb_prediction,
                    "dt_prediction": dt_prediction,
                }
            )
        except LangDetectException as e:
            logging.error(f"Language detection failed: {e}")
            return jsonify({"language": "Could not detect language"}), 500


if __name__ == "__main__":
    app.run(debug=True)
