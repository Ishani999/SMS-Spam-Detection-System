from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import pickle
import numpy as np

app = Flask(__name__)


model = pickle.load(open('rf_spam_classifier_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    data = [message]
    vect = vectorizer.transform(data).toarray()
    prediction = model.predict(vect)
    result = 'Spam' if prediction[0] == 1 else 'Ham'
    return render_template('index.html', prediction_text=f'The message is: {result}')
if __name__ == "__main__":
  app.run()  # Run the Flask development server