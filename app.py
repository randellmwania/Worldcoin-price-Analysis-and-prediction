from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('trained_model.h5')

# Define Flask app
app = Flask(__name__)

# Home route to render HTML form


@app.route('/')
def home():
    return render_template('index.html', prediction=None)

# Route to handle form submission and return predictions


@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    open_price = float(request.form['open_price'])
    high_price = float(request.form['high_price'])
    low_price = float(request.form['low_price'])
    volume = float(request.form['volume'])
    market_cap = float(request.form['market_cap'])

    # Create a DataFrame with the user inputs
    user_data = pd.DataFrame({
        'Open': [open_price],
        'High': [high_price],
        'Low': [low_price],
        'Volume': [volume],
        'Market Cap': [market_cap]
    })

    # Scale the user data
    scaler = MinMaxScaler()
    user_data_scaled = scaler.fit_transform(user_data)

    # Make prediction using the model
    prediction = model.predict(user_data_scaled)

    # Return the prediction to the home route
    return render_template('index.html', prediction=prediction[0][0])


if __name__ == '__main__':
    app.run(debug=True)
