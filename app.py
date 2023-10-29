from flask import Flask, request, jsonify, send_file
from statsmodels.tsa.api import SimpleExpSmoothing
from flask_cors import CORS, cross_origin
import pandas as pd
import os

app = Flask(__name__)
CORS(app, resources={r"/api/predict": {"origins": "http://localhost:5173", "origins": "https://batik-management-system.netlify.app"}})

# Direktori penyimpanan file
UPLOAD_FOLDER = 'files/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def home():
    return "Forecasting batik method"

@app.route('/api/predict', methods=['POST'])
@cross_origin()
def predict():
    # Get input data from the request
    
    # Check if a file is included in the request
    if 'file' in request.files:
        
        # print("file pada /predict: ", uploaded_file)
        files = request.files['file']

        # Membaca file CSV yang diunggah
        uploaded_data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], files.filename))
        uploaded_data['date'] = pd.to_datetime(uploaded_data['date'], format='%Y-%m')
        uploaded_data = uploaded_data.set_index('date')
        files.save(os.path.join(app.config['UPLOAD_FOLDER'], files.filename))
        
        # Create a model based on the uploaded data
        model = SimpleExpSmoothing(uploaded_data).fit(smoothing_level=0.8, optimized=False)

        # Perform prediction using the model
        forecast = model.forecast(steps=1)

        # Add labels for each forecasted month
        forecast_dates = pd.date_range(start=uploaded_data.index[-1], periods=1 + 1, freq='M')[1:]
        forecast.index = forecast_dates

        # Create a DataFrame with the forecast and month labels
        forecast_df = pd.DataFrame({'month': forecast.index.strftime('%B %Y'), 'forecast': forecast})

        # Convert the DataFrame to a JSON response
        response = forecast_df.to_json(orient='records')

        return response

    # If no file is included, return an appropriate response
    else:
        # print(file)
        print('Tidak ada file yang terdeteksi')
        return jsonify('Tidak ada file yang terdeteksi')

if __name__ == '__main__':
    app.run(debug=True)
