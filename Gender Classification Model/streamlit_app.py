import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

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
        
    # Encode company to numeric values
    label_encoder = LabelEncoder()
    df['company_encoded'] = label_encoder.fit_transform(df['company'])

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

    # Combine text embeddings with other numerical features
    numerical_features = ['code', 'company_encoded', 'age']
    X_numerical = df[numerical_features].values

    # Combine PCA-transformed text embeddings and numerical features
    X = np.hstack((text_embeddings_pca, X_numerical))

    # Scale the data using the same scaler used during training
    X = scaler.transform(X)

    # Make predictions using the trained Logistic Regression model
    y_pred = lr_model.predict(X)

    return y_pred[0]

# Streamlit application
def run():
    st.title("Gender Classification Model")

    # User input form
    usercode = st.number_input("Usercode (ID of Traveller)", min_value=0, max_value=1339, value=1234)
    company = st.selectbox("Company Name", ["Acme Factory", "Wonka Company", "Monsters CYA", "Umbrella LTDA", "4You"])
    name = st.text_input("Username", "Charlotte Johnson")
    age = st.slider("Traveller Age", min_value=21, max_value=65, value=30)

    # Create a dictionary with the input values
    data = {
        'code': usercode,
        'company': company,
        'name': name,
        'age': age
    }

    # Prediction button
    if st.button("Predict"):
        # Perform prediction using the data dictionary
        prediction = predict_price(data, logistic_model, pca_model, scaler_model)
        
        # Map the prediction to gender
        if prediction == 0:
            gender = 'female'
        else:
            gender = 'male'

        # Display the result
        st.subheader(f"The predicted gender is: {gender}")

if __name__ == "__main__":
    run()
