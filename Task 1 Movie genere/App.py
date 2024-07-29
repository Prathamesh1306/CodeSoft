from flask import Flask, request, render_template
import joblib

# Create Flask app
app = Flask(__name__)

# Load the model
model = joblib.load(r'C:\Users\Asus\OneDrive\Documents\Codesoft\Task 1\genre_classifier.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    description = None
    
    if request.method == 'POST':
        # Get the description from the request
        description = request.form['description']
        
        # Make prediction
        prediction = model.predict([description])[0]
        
        # Save the prediction
        with open('predictions.txt', 'a') as f:
            f.write(f"Description: {description}\nPredicted Genre: {prediction}\n\n")
    
    return render_template('index.html', description=description, genre=prediction)

if __name__ == '__main__':
    app.run(debug=True)
