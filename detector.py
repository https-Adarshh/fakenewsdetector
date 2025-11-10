import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import re

# Step 1: Connect to the live database
DB_USER = 'username'        # <-- replace with your MySQL username
DB_PASSWORD = 'password'    # <-- replace with your MySQL password
DB_HOST = 'localhost'
DB_NAME = 'fake_news_db'

DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
engine = create_engine(DB_URL)

# Step 2: Load datasets from the database table
query = "SELECT text, label FROM news_articles"
combined_data = pd.read_sql(query, engine)
combined_data = combined_data.sample(frac=1).reset_index(drop=True)

X = combined_data["text"]
y = combined_data["label"]

# Step 3: Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', text)
    return text

X = X.apply(preprocess_text)

# Step 4: Save all known preprocessed texts for verification (from database, stays updated)
known_texts = set(X)

# Step 5: Vectorizer and model training
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Step 7: Save model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Model and vectorizer saved successfully!")

# Step 8: Prediction function with live database verification
def predict_news(text):
    processed_text = preprocess_text(text)
    # Instead of checking in-memory, query the live DB for existence
    # (reconnect, for simulation or use a separate function in actual app)
    import pymysql
    connection = pymysql.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db=DB_NAME, charset='utf8mb4'
    )
    cursor = connection.cursor()
    sql = "SELECT 1 FROM news_articles WHERE text = %s"
    cursor.execute(sql, (processed_text,))
    found = cursor.fetchone()
    connection.close()
    
    if not found:
        return "Unverified"
    loaded_model = joblib.load("fake_news_model.pkl")
    loaded_vectorizer = joblib.load("vectorizer.pkl")
    vect_text = loaded_vectorizer.transform([processed_text])
    prediction = loaded_model.predict(vect_text)[0]
    return "True" if prediction == 1 else "Fake"

# Example usage:
# result = predict_news("Your news text here.")
# print(result)
