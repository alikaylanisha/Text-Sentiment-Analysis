from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import pickle

app = Flask(__name__)

# Initialize SnowballStemmer and a list of words to be removed
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))

# Define a function for preprocessing text
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
    text = re.sub("(\\d|\\W)+", " ", text)
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return " ".join(text)

# Load your trained model and TF-IDF vectorizer using pickle
with open('model_logreg.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

import warnings
warnings.filterwarnings('ignore', message='Trying to unpickle estimator .* from version .* when using version .*')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        # Preprocess the text
        text_processed = preprocess_text(text)
        # Vectorize the preprocessed text
        text_vectorized = tfidf_vectorizer.transform([text_processed])
        # Predict using the model
        prediction = model.predict(text_vectorized)[0]
        # Determine sentiment based on prediction
        if prediction == 0:
            sentiment = 'Negative'
        elif prediction == 1:
            sentiment = 'Neutral'
        else:
            sentiment = 'Positive'
        return render_template('index.html', text=text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
