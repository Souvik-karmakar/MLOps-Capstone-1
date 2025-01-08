# Example of a Flask endpoint for model inference
from flask import Flask, request, jsonify
import pickle  # For model serialization
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# # Initialize logging
# logging.basicConfig(
#     filename='app.log',
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# Initialize the SentenceTransformer model
model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')

# Load the trained classification model and scaler model
scaler_model = pickle.load(open("model/scaler_1.pkl", 'rb'))
pca_model = pickle.load(open("model/pca_1.pkl", 'rb'))
logistic_model = pickle.load(open("model/tuned_logistic_regression_model_1.pkl", 'rb'))

# Create a function for prediction
def predict_price(input_data, lr_model, pca, scaler):
    # Prepare the input data
    text_columns = ['name']

    # Initialize an empty DataFrame
    df = pd.DataFrame([input_data])
        
    
    # Encode userCode and company to numeric values
    label_encoder = LabelEncoder()

    df['company_encoded'] = label_encoder.fit_transform(df['company'])
    #df['gender_encoded'] = label_encoder.fit_transform(df['gender'])
    
    # Encode text-based columns and create embeddings
    for column in text_columns:
        df[column + '_embedding'] = df[column].apply(lambda text: model.encode(text))

    # Apply PCA separately to each text embedding column
    n_components = 23  # Adjust the number of components as needed
    text_embeddings_pca = np.empty((len(df), n_components * len(text_columns)))

    for i, column in enumerate(text_columns):
        embeddings = df[column + '_embedding'].values.tolist()
        embeddings_pca = pca.transform(embeddings)
        text_embeddings_pca[:, i * n_components:(i + 1) * n_components] = embeddings_pca

    # Combine text embeddings with other numerical features if available
    numerical_features = ['code','company_encoded','age']
    

    X_numerical = df[numerical_features].values

    # Combine PCA-transformed text embeddings and numerical features
    X = np.hstack((text_embeddings_pca, X_numerical))

    # Scale the data using the same scaler used during training
    X = scaler.transform(X)

    # Make predictions using the trained Linear Regression model
    y_pred = lr_model.predict(X)

    return y_pred[0]

# # Function to log user feedback
# def log_feedback(feedback_data):
#     try:
#         # Log the user feedback to a file
#         with open('feedback.log', 'a') as feedback_file:
#             feedback_file.write(f"{feedback_data}\n")
#         logger.info("User feedback logged successfully.")
#     except Exception as e:
#         logger.error(f"Error logging user feedback: {str(e)}")


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    return """<!DOCTYPE html>
<html>
<head>
    <title>Gender Classification Model</title>
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 50px;
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 20px 30px rgba(0, 0, 0, 0.1);
            text-align: center;
        }

        h1 {
            color: #009688;
            font-size: 40px;
            margin-bottom: 20px;
        }

        form {
            text-align: left;
        }

        input[type="text"],
        input[type="number"],
        select {
            width: 100%;
            padding: 14px;
            margin: 12px 0;
            border: 2px solid #009688;
            border-radius: 5px;
            font-size: 18px;
            background-color: #f9f9f9;
            color: #555;
            transition: border-color 0.3s ease;
        }

        input[type="text"]:focus,
        input[type="number"]:focus,
        select:focus {
            border-color: #00796b;
            outline: none;
        }

        input[type="submit"] {
            background-color: #009688;
            color: #ffffff;
            padding: 16px 32px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 20px;
            margin-top: 25px;
            transition: background-color 0.3s ease;
        }

        input[type="submit"]:hover {
            background-color: #00796b;
        }

        p#prediction {
            margin-top: 30px;
            font-size: 26px;
            color: #00796b;
        }

        label {
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Gender Classification Model</h1>
        <form action="/predict" method="POST">
            <label for="Username">Username:</label>
            <input type="text" name="Username" placeholder="Enter name of traveller" value="Charlotte Johnson">
            
            <label for="Usercode">Usercode:</label>
            <input type="number" name="Usercode" min="0.00" max="1339.00" placeholder="Enter the user id of traveller">

            <label for="Traveller_Age">Traveller Age:</label>
            <input type="number" name="Traveller_Age" min="21" max="65" placeholder="Enter the age of traveller">

            <label for="company_name">Company Name:</label>
            <select name="company_name">
                <option value="Acme Factory">Acme Factory</option>
                <option value="Wonka Company">Wonka Company</option>
                <option value="Monsters CYA">Monsters CYA</option>
                <option value="Umbrella LTDA">Umbrella LTDA</option>
                <option value="4You">4You</option>
            </select>

            <input type="submit" value="Predict">
        </form>
        <p id="prediction"></p>
    </div>
</body>
</html>"""

@app.route('/predict', methods=['POST'])
def index():
    if request.method == 'POST':
        # Get input data from the form
        usercode = request.form.get('Usercode')
        company = request.form.get('company_name')
        name = request.form.get('Username')
        age = request.form.get('Traveller_Age')

        # Create a dictionary to store the input data
        data = {
            'code': usercode,
            'company': company,
            'name': name,
            'age': age,
        }

        # Perform prediction using the custom_input dictionary
        prediction = predict_price(data, logistic_model, pca_model, scaler_model)
        
        if prediction == 0:
            gender = 'female'
        else:
            gender = 'male'
        
        prediction = str(gender)

        return jsonify({'prediction': prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)







