import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os

class HeadlineClassifier:
    """A classifier for predicting whether the article headline is clickbait or not"""
    def __init__(self, model_path: str = "headline_model.joblib", model_type: str = "logistic", random_state: int = 42):
        self.model_path = model_path
        self.model_type = model_type
        self.random_state = random_state
        self.pipeline = self._build_pipeline()
        self.is_trained = False

    def _build_pipeline(self):
        """Builds the pipeline based on selected model type"""
        if self.model_type == "logistic":
            classifier = LogisticRegression(solver="liblinear", random_state=self.random_state)
        elif self.model_type == "naive_bayes":
            classifier = MultinomialNB()
        elif self.model_type == "random_forest":
            classifier = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        elif self.model_type == "svm":
            classifier = LinearSVC(random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported model_type '{self.model_type}'.")

        return Pipeline([
            ("vectorizer", TfidfVectorizer(stop_words="english", max_df=0.9)),
            ("classifier", classifier)
        ])
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Loads data from .csv file containing 'headline' and 'clickbait' columns
        Returns pandas DataFrame object
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found")

        headlines_df = pd.read_csv(filepath)
        if "headline" not in headlines_df.columns or "clickbait" not in headlines_df.columns:
            raise ValueError("CSV file does not containt 'headline' or 'clickbait' column")

        return headlines_df

    def train(self, data: pd.DataFrame):
        """Trains the classifier based on the provided data"""
        X = data["headline"]
        y = data["clickbait"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=self.random_state)

        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.pipeline.predict(X_test)
        print("Model performance on the testing set:")
        print(classification_report(y_test, y_pred))

    def save_model(self):
        """Saves the model for later use"""
        if not self.is_trained:
            raise RuntimeError("Model is not trained, train the model before saving")
        joblib.dump(self.pipeline, self.model_path)
        print(f"Model has been saved in {self.model_path}")

    def load_model(self):
        """Loads trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"{self.model_path} not found")
        self.pipeline = joblib.load(self.model_path)
        self.is_trained = True
        print(f"Model {self.model_path} has been loaded")

    def predict(self, headlines: list[str]) -> list[bool]:
        """Predicts whether headline is clickbait or not"""
        if not self.is_trained:
            raise RuntimeError("Model is not trained, train the model before the usage")
        return self.pipeline.predict(headlines).tolist()

if __name__ == "__main__":
    classifier = HeadlineClassifier()

    try:
        df = classifier.load_data("data/headline_clickbait.csv")
        classifier.train(df)
        classifier.save_model()
    except Exception as e:
        print(f"Error: {e}")