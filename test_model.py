# pip install scikit-learn spacy joblib
# python -m spacy download en_core_web_sm

import joblib
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Sample dataset (clickbait = 1, non-clickbait = 0)
headlines = [
    ("You Won't Believe What Happened Next!", 1),
    ("10 Secrets About Weight Loss That Doctors Won't Tell You", 1),
    ("Scientists Discover New Species in the Amazon", 0),
    ("How to Make a Million Dollars in One Month!", 1),
    ("Government Announces New Tax Policies", 0),
    ("This Simple Trick Will Change Your Life!", 1),
    ("Breaking: Major Earthquake Hits California", 0),
    ("Top 5 Foods That Will Help You Lose Weight Fast", 1),
    ("NASA Releases Stunning New Images of Mars", 0),
    ("The One Thing You Should Never Do in an Interview", 1)
]

# Preprocess function using SpaCy
nlp = spacy.load('en_core_web_sm')

def preprocess(text):
    doc = nlp(text.lower())
    return " ".join(token.lemma_ for token in doc if token.is_alpha and not token.is_stop)

# Prepare data
texts, labels = zip(*headlines)
texts = [preprocess(text) for text in texts]

# Convert text to numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, "clickbait_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved successfully.")
