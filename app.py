import re
from textblob import TextBlob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import joblib
import os

class SentimentAnalyzer:
    def __init__(self, model_path='models/sentiment_model.pkl', vectorizer_path='models/tfidf_vectorizer.pkl'):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
        
        # Try to load existing model and vectorizer
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.model_loaded = True
        except:
            self.model = None
            self.vectorizer = None
            self.model_loaded = False
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    def train_model(self, data_path='data/product_reviews.csv'):
        """Train the sentiment analysis model"""
        try:
            # Load dataset
            df = pd.read_csv(data_path)
            
            # Preprocess text
            df['cleaned_text'] = df['text'].apply(self.preprocess_text)
            
            # Split data
            X = df['cleaned_text']
            y = df['sentiment']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Vectorize text
            self.vectorizer = TfidfVectorizer(max_features=5000)
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # Train model
            self.model = LinearSVC()
            self.model.fit(X_train_vec, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model trained with accuracy: {accuracy:.2f}")
            
            # Save model and vectorizer
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.vectorizer, self.vectorizer_path)
            self.model_loaded = True
            
            return accuracy
        except Exception as e:
            print(f"Error in training: {e}")
            return None
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of the given text"""
        if not self.model_loaded:
            print("Model not loaded. Please train or load a model first.")
            return None
        
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Vectorize text
        text_vec = self.vectorizer.transform([cleaned_text])
        
        # Predict sentiment
        prediction = self.model.predict(text_vec)[0]
        
        # Get confidence score using TextBlob as fallback
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Combine results
        if prediction == 'positive' and polarity > 0.1:
            confidence = max(0.5, polarity)
            return {'sentiment': 'positive', 'confidence': confidence}
        elif prediction == 'negative' and polarity < -0.1:
            confidence = max(0.5, abs(polarity))
            return {'sentiment': 'negative', 'confidence': confidence}
        else:
            # If model and TextBlob disagree, use TextBlob with lower confidence
            if polarity > 0:
                return {'sentiment': 'positive', 'confidence': 0.5 + polarity/2}
            else:
                return {'sentiment': 'negative', 'confidence': 0.5 + abs(polarity)/2}

def main():
    analyzer = SentimentAnalyzer()
    
    # Train model if not already loaded
    if not analyzer.model_loaded:
        print("No trained model found. Training a new model...")
        analyzer.train_model()
    
    print("\nProduct Review Sentiment Analyzer")
    print("Enter 'quit' to exit\n")
    
    while True:
        text = input("Enter your product review/comment: ")
        
        if text.lower() == 'quit':
            break
        
        result = analyzer.analyze_sentiment(text)
        
        if result:
            print(f"\nSentiment: {result['sentiment'].upper()}")
            print(f"Confidence: {result['confidence']*100:.1f}%")
            print("-" * 30)

if __name__ == "__main__":
    main()