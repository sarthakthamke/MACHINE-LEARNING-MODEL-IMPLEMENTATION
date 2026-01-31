import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import re
import string

def preprocess_text(text):
    """
    Preprocess the text data: lowercase, remove punctuation, etc.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def load_and_preprocess_data(file_path):
    """
    Load the dataset and preprocess the text.
    """
    try:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',', 1)  # Split only on first comma
                if len(parts) == 2:
                    label, text = parts
                    data.append({'label': label, 'text': text})
        
        df = pd.DataFrame(data)
        if df.empty:
            raise ValueError("Dataset is empty or not in expected format.")
        
        print("Dataset loaded:")
        print(df.head())
        print(f"\nDataset shape: {df.shape}")
        print(f"Label distribution:\n{df['label'].value_counts()}")

        # Preprocess text
        df['processed_text'] = df['text'].apply(preprocess_text)

        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file_path}' not found. Please ensure the dataset file exists.")
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")

def train_spam_classifier(df):
    """
    Train a spam classification model using SVM.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
    )

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"\nTF-IDF feature matrix shape: {X_train_tfidf.shape}")

    # Train the model
    model = SVC(kernel='linear', probability=True, class_weight='balanced')
    model.fit(X_train_tfidf, y_train)

    # Make predictions
    y_pred = model.predict(X_test_tfidf)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the model and vectorizer
    joblib.dump(model, 'spam_classifier_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("\nModel and vectorizer saved successfully!")

    return model, vectorizer

def load_model_and_predict(text):
    """
    Load the saved model and vectorizer to make predictions.
    """
    try:
        model = joblib.load('spam_classifier_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        
        prediction, spam_prob = predict_spam(model, vectorizer, text)
        return prediction, spam_prob
    except FileNotFoundError:
        raise FileNotFoundError("Model or vectorizer file not found. Please train the model first.")

def predict_spam(model, vectorizer, text):
    """
    Predict if a given text is spam or ham.
    """
    # Preprocess the text
    processed_text = preprocess_text(text)
    # Vectorize
    text_tfidf = vectorizer.transform([processed_text])
    # Predict
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    if 'spam' in model.classes_:
        spam_prob = probabilities[list(model.classes_).index('spam')]
    else:
        spam_prob = 0.0  # Default if 'spam' not in classes

    return prediction, spam_prob

def main():
    """
    Main function to run the spam detection model.
    """
    data_file = "spam_dataset.csv"

    try:
        # Load and preprocess data
        df = load_and_preprocess_data(data_file)

        # Train the model
        model, vectorizer = train_spam_classifier(df)

        # Test predictions on sample texts
        test_texts = [
            "Congratulations! You've won a free iPhone. Click here to claim.",
            "Hey, are we still meeting for lunch tomorrow?",
            "URGENT: Your account has been suspended. Call now to reactivate.",
            "Thanks for your help with the project. It was great working with you."
        ]

        print("\n" + "="*50)
        print("TEST PREDICTIONS")
        print("="*50)

        for text in test_texts:
            prediction, spam_prob = predict_spam(model, vectorizer, text)
            print(f"\nText: {text}")
            print(f"Prediction: {prediction.upper()}")
            print(f"Spam Probability: {spam_prob:.4f}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()