from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import pickle
import logging
import os

# Initialize the Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained models and dependencies
try:
    scaler_model = pickle.load(open("model/scaler.pkl", 'rb'))
    pca_model = pickle.load(open("model/pca.pkl", 'rb'))
    logistic_model = pickle.load(open("model/tuned_logistic_regression_model.pkl", 'rb'))
    logging.info("Models loaded successfully.")
except FileNotFoundError as e:
    logging.error(f"Error loading models: {e}")
    exit(1)

# Load SentenceTransformer
text_model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')

# Initialize LabelEncoder
gender_encoder = LabelEncoder()
gender_encoder.classes_ = np.array(['male', 'female'])  # Ensure consistency

@app.route('/')
def index():
    """Render the main form for user input."""
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict_price():
    """Predict gender based on input data."""
    try:
        # Extract input data from the form
        usercode = request.form.get('usercode', type=int)
        age = request.form.get('age', type=int)
        name = request.form.get('name', type=str)
        company = request.form.get('company', type=str)

        # Validate inputs
        if not all([usercode, age, name, company]):
            return jsonify({'error': 'All fields are required!'}), 400

        # Prepare input data
        user_data = pd.DataFrame([{
            'code': usercode,
            'age': age,
            'name': name,
            'company': company
        }])

        # Encode categorical columns
        company_encoded = pd.get_dummies(user_data['company'], prefix='company')

        # Transform name to embeddings
        user_data['name_embedding'] = user_data['name'].apply(lambda x: text_model.encode(x))
        embeddings_pca = pca_model.transform(np.vstack(user_data['name_embedding']))

        # Combine embeddings and numerical features
        numerical_features = user_data[['code', 'age']].values
        combined_features = np.hstack((embeddings_pca, numerical_features, company_encoded.values))

        # Scale input data
        scaled_features = scaler_model.transform(combined_features)

        # Make predictions
        predictions = logistic_model.predict(scaled_features)
        predicted_gender = gender_encoder.inverse_transform(predictions)

        return jsonify({
            'usercode': usercode,
            'name': name,
            'predicted_gender': predicted_gender[0]
        })

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction. Please try again later.'}), 500

if __name__ == '__main__':
    # Run the app
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
