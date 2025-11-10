from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import joblib
import re

app = Flask(__name__)
CORS(app)

# MySQL Database Config: UPDATE with your own credentials!
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://username:password@localhost/fake_news_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class NewsArticle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, unique=True, nullable=False)
    label = db.Column(db.Integer, nullable=False)

try:
    model = joblib.load('fake_news_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except FileNotFoundError:
    print("Model or vectorizer not found. Please run detector.py first.")
    exit()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', text)
    return text

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['text']
        cleaned_text = preprocess_text(data)

        # 1. Check in database
        article = NewsArticle.query.filter_by(text=cleaned_text).first()
        if not article:
            return jsonify({'prediction': 2})  # Unverified

        # 2. If found, run the prediction and threshold logic
        processed_text = vectorizer.transform([cleaned_text])
        proba = model.predict_proba(processed_text)[0]
        pred = model.predict(processed_text)[0]
        if max(proba) < 0.75:
            prediction = 2  # Unverified (low confidence)
        else:
            prediction = pred

        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
