from flask import Flask, request, jsonify, render_template
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Ensure the results are consistent by setting seed
DetectorFactory.seed = 0

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the dataset (replace with your actual path)
df = pd.read_csv("C:/Users/Suhas sattigeri/Desktop/Mini P/data/cleaned_dataset1.csv")

# Preprocess the data (adjust based on your needs)
df["Text"] = df["Text"].str.lower().str.replace("[^\w\s]", "", regex=True)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Split the dataset into training and test sets first
X = df["Text"]
y = df["Language"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the vectorizer only on the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the test data using the already fitted vectorizer
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Train a Decision Tree classifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_tfidf, y_train)

# Evaluate the models on the test set
nb_predictions = nb_model.predict(X_test_tfidf)
dt_predictions = dt_model.predict(X_test_tfidf)

nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_precision = precision_score(y_test, nb_predictions, average="weighted")
nb_recall = recall_score(y_test, nb_predictions, average="weighted")
nb_f1 = f1_score(y_test, nb_predictions, average="weighted")

dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_precision = precision_score(y_test, dt_predictions, average="weighted")
dt_recall = recall_score(y_test, dt_predictions, average="weighted")
dt_f1 = f1_score(y_test, dt_predictions, average="weighted")

# Print detailed classification reports
print("Naive Bayes Classification Report:")
print(classification_report(y_test, nb_predictions))

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, dt_predictions))

@app.route("/")
def main():
    return render_template("main.html")

@app.route("/classify")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        text = request.get_json().get("text", "")
        if not text:
            return jsonify({"error": "Missing or empty text field"}), 400
        try:
            language = detect(text)
            text_tfidf = vectorizer.transform([text])
            nb_prediction = nb_model.predict(text_tfidf)[0]
            dt_prediction = dt_model.predict(text_tfidf)[0]
            nb_confidence = nb_model.predict_proba(text_tfidf)[0][
                nb_model.classes_.tolist().index(nb_prediction)
            ]
            dt_confidence = None
            if hasattr(dt_model, 'predict_proba'):
                dt_confidence = dt_model.predict_proba(text_tfidf)[0][
                    dt_model.classes_.tolist().index(dt_prediction)
                ]
            return jsonify(
                {
                    "language": language,
                    "nb_prediction": nb_prediction,
                    "dt_prediction": dt_prediction,
                    "nb_confidence": nb_confidence * 100,
                    "dt_confidence": dt_confidence * 100 if dt_confidence is not None else None,
                }
            )
        except LangDetectException as e:
            logging.error(f"Language detection failed: {e}")
            return jsonify({"language": "Could not detect language"}), 500

@app.route("/results")
def results():
    return render_template(
        "results.html",
        nb_accuracy=nb_accuracy,
        nb_precision=nb_precision,
        nb_recall=nb_recall,
        nb_f1=nb_f1,
        dt_accuracy=dt_accuracy,
        dt_precision=dt_precision,
        dt_recall=dt_recall,
        dt_f1=dt_f1,
    )

if __name__ == "__main__":
    app.run(debug=True)

print(df['Language'].value_counts())
