from os import mkdir
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score # Import f1_score
import joblib
import os
import logging_config
import logging
from config import GENERAL_CONFIG

logger = logging.getLogger(__name__)

class HeadlineClassifier:
    """A classifier for predicting whether the article headline is clickbait or not"""
    def __init__(self, model_path: str = "models/headline_models", model_type: str = "logistic", random_state: int = GENERAL_CONFIG["seed"]):
        if model_path.endswith(".joblib"):
            self.model_path = model_path
        else:
            self.model_path = os.path.join(model_path, model_type + ".joblib")
        self.model_type = model_type
        self.random_state = random_state
        self.pipeline = self._build_pipeline()
        self.is_trained = False

    def predict_proba(self, headlines: list[str]) -> list[float]:
        """Predicts the probability of headlines being clickbait."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained, train the model before the usage")

        # predict_proba returns probabilities for each class [class_0, class_1]
        # We need to find the index of the "clickbait" class (usually 1)
        try:
            # Assuming the positive class (clickbait) is labeled as 1
            clickbait_class_index = list(self.pipeline.classes_).index(1)
            probabilities = self.pipeline.predict_proba(headlines)[:, clickbait_class_index]
            return probabilities.tolist()
        except Exception as e:
            logging.error(f"Error getting prediction probability: {e}. Defaulting to standard predict.")
            # Fallback for models that might not have predict_proba or if classes are unexpected
            return [float(p) for p in self.predict(headlines)]

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

    def train(self, X_train: pd.Series, y_train: pd.Series):
        """Trains the classifier based on the provided training data"""
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        logging.info(f"Model training completed for {self.model_type}.")

    def test(self, X_test: pd.Series, y_test: pd.Series):
        """Tests the classifier on the provided test data and prints a classification report.
           Returns the F1-score for the positive class (1).
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained, train the model before testing")

        y_pred = self.pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        logging.info(f"Performance for {self.model_type} model")
        logging.info(f"\n{classification_report(y_test, y_pred)}")
        logging.info(GENERAL_CONFIG["separator"]) # Separator

        # Return F1-score for the positive class (assuming 1 is the positive class)
        return report['1']['f1-score']


    def save_model(self):
        """Saves the model for later use"""
        if not self.is_trained:
            raise RuntimeError("Model is not trained, train the model before saving")
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.mkdir(os.path.dirname(self.model_path))
        joblib.dump(self.pipeline, self.model_path)
        logging.info(f"Model has been saved in {self.model_path}")

    def load_model(self):
        """Loads trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"{self.model_path} not found")
        self.pipeline = joblib.load(self.model_path)
        self.is_trained = True
        logging.info(f"Model {self.model_path} has been loaded")

    def predict(self, headlines: list[str]) -> list[bool]:
        """Predicts whether headline is clickbait or not"""
        if not self.is_trained:
            raise RuntimeError("Model is not trained, train the model before the usage")
        return self.pipeline.predict(headlines).tolist()

if __name__ == "__main__":

    # Load data once
    data_filepath = "data/headline_clickbait.csv" # Ensure this file exists
    try:
        classifier_data_loader = HeadlineClassifier() # Temporary instance to load data
        df = classifier_data_loader.load_data(data_filepath)
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Error loading data: {e}")
        exit()

    X = df["headline"]
    y = df["clickbait"]

    # Perform the train-test split once
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=GENERAL_CONFIG["seed"])

    # Define the models to test
    model_types_to_test = ["logistic", "naive_bayes", "random_forest", "svm"]

    best_score = -1 # Initialize with a low score
    best_classifier: HeadlineClassifier = None # To store the best model instance
    best_model_type = ""

    for model_type in model_types_to_test:
        logging.info(f"\n--- Processing {model_type} Model ---")
        classifier = HeadlineClassifier(model_type=model_type)

        # Train the model
        classifier.train(X_train, y_train)

        # Test the model and get the F1-score
        current_score = classifier.test(X_test, y_test)

        # Compare and update if current model is better
        if current_score > best_score:
            best_score = current_score
            best_classifier = classifier
            best_model_type = model_type
            logging.info(f"New best model found: {best_model_type} with F1-score: {best_score:.4f}")

    if best_classifier:
        logging.info(f"Saving the best model: {best_model_type} with F1-score: {best_score:.4f}")
        # Update the model_path for the best classifier to ensure it saves with its name
        best_classifier.model_path = os.path.join(os.path.dirname(best_classifier.model_path), best_model_type + "_best.joblib")
        best_classifier.save_model()
    else:
        logging.warning("No best model found. This might happen if no models were trained or an error occurred.")