from flask import Flask, request, jsonify, render_template
import joblib

# Load the model and vectorizer
model = joblib.load(r'C:\Users\Asus\OneDrive\Documents\Codesoft\3. Spam detection\spam_classifier.pkl')
vectorizer = joblib.load(r'C:\Users\Asus\OneDrive\Documents\Codesoft\3. Spam detection\tf-idfvector.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the message from the POST request
    message = request.form['message']
    
    # Transform the message using the TF-IDF vectorizer
    message_tfidf = vectorizer.transform([message])
    
    # Predict the label using the loaded model
    prediction = model.predict(message_tfidf)
    
    # Convert the prediction to a readable label
    label = 'spam' if prediction[0] == 1 else 'ham'
    
    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)
