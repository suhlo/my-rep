import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import joblib
import spacy  # type: ignore

# Load the spaCy model for tokenization and entity recognition
nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):
    # Tokenize the text
    tokens = [token.text for token in nlp(text)]
    # Remove stopwords and punctuation
    tokens = [
        token
        for token in tokens
        if token not in nlp.Defaults.stop_words and token.isalpha()
    ]
    # Join the tokens back into a string
    text = " ".join(tokens)
    return text


# Sample data
data = {
    "text": [
        "I love this movie",
        "I hate this movie",
        "This movie is great",
        "This movie is awful",
    ],
    "label": [1, 0, 1, 0],
}

df = pd.DataFrame(data)

# Preprocess the text data
df["text"] = df["text"].apply(preprocess_text)

# Vectorize the text data
X = df["text"]
y = df["label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a pipeline with a vectorizer and a classifier
model = make_pipeline(TfidfVectorizer(), LogisticRegression())

# Train the model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "text_classification_model.pkl")
