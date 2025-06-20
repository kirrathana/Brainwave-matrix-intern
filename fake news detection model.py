# Fake News Detection Model 
import os
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Define file path - using absolute path for reliability
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'fake_news_dataset.csv')

# Create dataset if it doesn't exist
if not os.path.exists(csv_path):
    print("Creating fake_news_dataset.csv...")
    data = {
        "title": [
            "NASA Announces Moon Made of Cheese",
            "Regular Exercise Found to Improve Heart Health",
            "Aliens Land in New York, Government Confirms",
            "WHO Confirms COVID-19 Vaccines Are Safe",
            "Bleach Drinking Cures COVID-19, Claims Doctor",
            "Scientists Reach Consensus on Climate Change",
            "5G Networks Causing Coronavirus Outbreak",
            "Vegetables Reduce Chronic Disease Risk"
        ],
        "text": [
            "NASA scientists have made the shocking discovery that the moon is actually composed primarily of green cheese.",
            "A comprehensive new study shows that just 30 minutes of daily exercise can significantly improve cardiovascular health.",
            "Government officials have confirmed that extraterrestrial beings landed in Central Park yesterday evening.",
            "The World Health Organization has released a statement confirming that all approved COVID-19 vaccines are safe.",
            "Controversial physician claims that drinking diluted bleach can cure COVID-19 within hours.",
            "Over 99% of peer-reviewed scientific papers agree that climate change is real and primarily caused by human activity.",
            "A leaked study suggests that 5G wireless networks are directly responsible for the spread of coronavirus.",
            "New research indicates that consuming vegetables daily can reduce chronic disease risk by up to 40%."
        ],
        "label": [1, 0, 1, 0, 1, 0, 1, 0]  # 1=Fake, 0=Real
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"Dataset created at: {csv_path}")
else:
    print(f"Using existing dataset at: {csv_path}")
    df = pd.read_csv(csv_path)

# Data cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Preprocess data
print("\nCleaning text data...")
df['clean_text'] = df['text'].apply(clean_text)

# Split data
X = df['clean_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature extraction
print("Creating TF-IDF features...")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train model
print("Training model...")
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Evaluate
print("Evaluating model...")
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("\nRESULTS:")
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(cr)

# Prediction function
def predict_news(text):
    cleaned_text = clean_text(text)
    vectorized_text = tfidf.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    return "Fake News" if prediction[0] == 1 else "Real News"

# Test predictions
print("\nTest Predictions:")
test_samples = [
    "Chocolate found to cure all diseases",
    "Study confirms benefits of regular exercise",
    "Government hiding alien technology",
    "Washing hands prevents illness"
]

for i, text in enumerate(test_samples, 1):
    print(f"\nExample {i}:")
    print(f"Text: {text}")
    print(f"Prediction: {predict_news(text)}")