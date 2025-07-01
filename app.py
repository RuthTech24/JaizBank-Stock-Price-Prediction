# app.py
from flask import Flask, request, render_template
import joblib
import numpy as np

# Load the trained model
model = joblib.load('JaizBank_ProvantageNGX-AI_model.pkl')

# Initialize the Flask app
app = Flask(__name__)

# Home route: show form and (optional) result
@app.route('/')
def index():
    return render_template('index.html', predicted_price=None)

# Prediction route: handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from form
        Price = float(request.form['Price'])
        MA_5 = float(request.form['MA_5'])
        MA_10 = float(request.form['MA_10'])
        Lag_1 = float(request.form['Lag_1'])
        Lag_2 = float(request.form['Lag_2'])
        Daily_Return = float(request.form['Daily_Return'])

        # Structure features into correct format for model
        features = np.array([[Price, MA_5, MA_10, Lag_1, Lag_2, Daily_Return]])

        # Make prediction
        prediction = model.predict(features)
        predicted_price = round(prediction[0], 4)

        # Render form again with prediction result
        return render_template('index.html', predicted_price=predicted_price)

    except Exception as e:
        # In case of error (e.g., input not convertible to float), show error on page
        return render_template('index.html', predicted_price=f"Error: {str(e)}")

# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)
