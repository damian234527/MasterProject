"""A scikit-learn based classifier for headline clickbait detection.

This module provides the `HeadlineClassifier` class, which encapsulates a
scikit-learn pipeline for vectorizing text and training a classifier. It supports
various classification models like Logistic Regression, SVM, and Naive Bayes.
The main script functionality allows for training, evaluating, and saving the
best performing model based on F1-score.
"""
from os import mkdir
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
import joblib
import os
import logging
from config import GENERAL_CONFIG

logger = logging.getLogger(__name__)


class HeadlineClassifier:
    """A classifier for predicting whether an article headline is clickbait.

    This class wraps a scikit-learn `Pipeline` that consists of a `TfidfVectorizer`
    and a classifier. It provides a unified interface for training, testing,
    predicting, and saving/loading the model.

    Attributes:
        model_path (str): The full file path for saving or loading the model.
        model_type (str): The type of classifier to use (e.g., 'logistic').
        random_state (int): The random seed for reproducibility.
        pipeline (sklearn.pipeline.Pipeline): The underlying scikit-learn
            pipeline object.
        is_trained (bool): A flag indicating whether the model has been trained.
    """

    def __init__(self, model_path: str = "models/headline_models", model_type: str = "logistic",
                 random_state: int = GENERAL_CONFIG["seed"]):
        """Initializes the HeadlineClassifier.

        Args:
            model_path (str, optional): The directory or full path to the model
                file. If a directory is given, the model type is appended.
                Defaults to "models/headline_models".
            model_type (str, optional): The type of classifier to use.
                Supported types: 'logistic', 'naive_bayes', 'random_forest',
                'svm', 'sgd', 'passive_aggressive'. Defaults to "logistic".
            random_state (int, optional): The random seed for classifiers that
                support it. Defaults to the value in `GENERAL_CONFIG`.
        """
        # Construct the full model path if a directory is provided.
        if model_path.endswith(".joblib"):
            self.model_path = model_path
        else:
            self.model_path = os.path.join(model_path, model_type + ".joblib")
        self.model_type = model_type
        self.random_state = random_state
        self.pipeline = self._build_pipeline()
        self.is_trained = False

    def predict_proba(self, headlines: list[str]) -> list[float]:
        """Predicts the probability of each headline being clickbait.

        Args:
            headlines (list[str]): A list of headline strings to predict.

        Returns:
            A list of floating-point probabilities, where each probability
            corresponds to the 'clickbait' class (1).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained, train the model before the usage")

        try:
            # `predict_proba` returns probabilities for all classes [class_0, class_1].
            # We need the index of the positive class (1 for clickbait).
            clickbait_class_index = list(self.pipeline.classes_).index(1)
            probabilities = self.pipeline.predict_proba(headlines)[:, clickbait_class_index]
            return probabilities.tolist()
        except Exception as e:
            # Fallback for models that do not support `predict_proba` (e.g., LinearSVC).
            logger.error(f"Error getting prediction probability: {e}. Defaulting to standard predict.")
            return [float(p) for p in self.predict(headlines)]

    def _build_pipeline(self):
        """Builds the scikit-learn pipeline based on the selected model type.

        Returns:
            A scikit-learn `Pipeline` object.

        Raises:
            ValueError: If an unsupported `model_type` is specified.
        """
        # Select the classifier based on the model_type attribute.
        if self.model_type == "logistic":
            classifier = LogisticRegression(solver="liblinear", random_state=self.random_state)
        elif self.model_type == "naive_bayes":
            classifier = MultinomialNB()
        elif self.model_type == "random_forest":
            classifier = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        elif self.model_type == "svm":
            classifier = LinearSVC(random_state=self.random_state)
        elif self.model_type == "sgd":
            classifier = SGDClassifier(random_state=self.random_state, loss="hinge")
        elif self.model_type == "passive_aggressive":
            classifier = PassiveAggressiveClassifier(random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported model_type '{self.model_type}'.")

        # Assemble the vectorizer and classifier into a pipeline.
        return Pipeline([
            ("vectorizer", TfidfVectorizer(stop_words="english", max_df=0.9)),
            ("classifier", classifier)
        ])

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Loads data from a CSV file.

        The CSV file must contain 'headline' and 'clickbait' columns.

        Args:
            filepath (str): The path to the CSV file.

        Returns:
            A pandas DataFrame containing the loaded data.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the CSV is missing the required columns.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found")

        headlines_df = pd.read_csv(filepath)
        if "headline" not in headlines_df.columns or "clickbait" not in headlines_df.columns:
            raise ValueError("CSV file does not contain 'headline' or 'clickbait' column")

        return headlines_df

    def train(self, X_train: pd.Series, y_train: pd.Series):
        """Trains the classifier on the provided training data.

        Args:
            X_train (pd.Series): A pandas Series of headline strings.
            y_train (pd.Series): A pandas Series of corresponding labels (0 or 1).
        """
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        logger.info(f"Model training completed for {self.model_type}.")

    def test(self, X_test: pd.Series, y_test: pd.Series):
        """Tests the classifier and returns the F1-score for the positive class.

        Also logs a full classification report.

        Args:
            X_test (pd.Series): A pandas Series of headline strings for testing.
            y_test (pd.Series): A pandas Series of true labels for the test data.

        Returns:
            The F1-score for the positive class (label 1).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained, train the model before testing")

        y_pred = self.pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        logger.info(f"Performance for {self.model_type} model")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        logger.info(GENERAL_CONFIG["separator"])

        # Return the F1-score for the clickbait class (1).
        return report['1']['f1-score']

    def save_model(self):
        """Saves the trained pipeline to a file using joblib.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained, train the model before saving")
        # Create the directory if it doesn't exist.
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.mkdir(os.path.dirname(self.model_path))
        joblib.dump(self.pipeline, self.model_path)
        logger.info(f"Model has been saved in {self.model_path}")

    def load_model(self):
        """Loads a pre-trained pipeline from a file.

        Raises:
            FileNotFoundError: If the model file does not exist at `self.model_path`.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"{self.model_path} not found")
        self.pipeline = joblib.load(self.model_path)
        self.is_trained = True
        logger.info(f"Model {self.model_path} has been loaded")

    def predict(self, headlines: list[str]) -> list[bool]:
        """Predicts whether each headline in a list is clickbait.

        Args:
            headlines (list[str]): A list of headline strings to classify.

        Returns:
            A list of booleans, where True indicates a clickbait prediction.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained, train the model before the usage")
        return self.pipeline.predict(headlines).tolist()


if __name__ == "__main__":
    import logging_config

    # Load the dataset once.
    data_filepath = "data/headline_clickbait.csv"
    try:
        classifier_data_loader = HeadlineClassifier()
        df = classifier_data_loader.load_data(data_filepath)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error loading data: {e}")
        exit()

    X = df["headline"]
    y = df["clickbait"]

    # Split the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=GENERAL_CONFIG["seed"])

    # Define the list of model types to train and evaluate.
    model_types_to_test = ["logistic", "naive_bayes", "random_forest", "svm", "sgd", "passive_aggressive"]

    best_score = -1
    best_classifier: HeadlineClassifier = None
    best_model_type = ""

    # Iterate through each model type, train it, and evaluate its performance.
    for model_type in model_types_to_test:
        logger.info(f"\n--- Processing {model_type} Model ---")
        classifier = HeadlineClassifier(model_type=model_type)

        # Train the model on the training data.
        classifier.train(X_train, y_train)

        # Test the model and get its F1-score.
        current_score = classifier.test(X_test, y_test)
        logger.info(f"Score for {model_type}: {current_score}")

        # If the current model is the best so far, save it.
        if current_score > best_score:
            best_score = current_score
            best_classifier = classifier
            best_model_type = model_type
            logger.info(f"New best model found: {best_model_type} with F1-score: {best_score:.4f}")

    # After checking all models, save the best one to a file.
    if best_classifier:
        logger.info(f"Saving the best model: {best_model_type} with F1-score: {best_score:.4f}")
        # Ensure the model is saved with the correct filename.
        best_classifier.model_path = os.path.join(os.path.dirname(best_classifier.model_path),
                                                  best_model_type + ".joblib")
        best_classifier.save_model()
    else:
        logger.warning("No best model found. This might happen if no models were trained or an error occurred.")