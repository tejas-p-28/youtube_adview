import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import datetime

# Load the model
model = joblib.load('decisiontree_youtubeadview.pkl')

# Initialize encoders and scaler (these should ideally be saved with the model)
category_encoder = LabelEncoder()
published_encoder = LabelEncoder()
scaler = MinMaxScaler()

# Category mapping
category_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8}

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        views = int(request.form['views'])
        likes = int(request.form['likes'])
        dislikes = int(request.form['dislikes'])
        comment = int(request.form['comment'])
        published = request.form['published']
        duration = int(request.form['duration'])
        category = request.form['category']
        
        # Create a DataFrame with the input data
        data = pd.DataFrame({
            'views': [views],
            'likes': [likes],
            'dislikes': [dislikes],
            'comment': [comment],
            'published': [published],
            'duration': [duration],
            'category': [category]
        })
        
        # Apply the same preprocessing as in training
        
        # 1. Convert category to numeric
        data['category'] = data['category'].map(category_mapping)
        
        # 2. Convert published date to numeric (using LabelEncoder approach)
        # For simplicity, we'll use the date as a numeric value (days since epoch)
        data['published'] = pd.to_datetime(data['published']).astype(np.int64) // 10**9
        
        # 3. Ensure all numeric columns are properly typed
        data['views'] = pd.to_numeric(data['views'])
        data['comment'] = pd.to_numeric(data['comment'])
        data['likes'] = pd.to_numeric(data['likes'])
        data['dislikes'] = pd.to_numeric(data['dislikes'])
        data['duration'] = pd.to_numeric(data['duration'])
        
        # 4. Apply MinMaxScaler (using the same scaling approach as training)
        # Note: In a production environment, you should save and load the fitted scaler
        features = data.values
        features_scaled = scaler.fit_transform(features)
        
        # 5. Make prediction
        prediction = model.predict(features_scaled)
        
        # Format the prediction
        prediction_value = int(prediction[0])
        
        return render_template('index.html', 
                             prediction_text=f'The predicted adview is {prediction_value:,} views')
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)