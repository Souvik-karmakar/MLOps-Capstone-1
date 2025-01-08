# Import required libraries
import pandas as pd
import pickle
import streamlit as st

# Function to predict the price
def predict_price(input_data, model, scaler):
    # Prepare the input data
    df_input2 = pd.DataFrame([input_data])  # Initialize an empty DataFrame
    X = df_input2  # Independent features 

    # Scale the data using the same scaler used during training (Standard Scaler)
    X = scaler.transform(X)

    # Make predictions using the trained model
    y_prediction = model.predict(X)
    
    return y_prediction[0]


# Load models
scaler_model = pickle.load(open("model/scaling_1.pkl", 'rb'))
rf_model = pickle.load(open("model/rf_model.pkl", 'rb'))

# Streamlit App
def main():
    st.title('Flight Price Prediction')

    # Input fields in the Streamlit app
    boarding = st.selectbox('Select Boarding City', ['Florianopolis (SC)', 'Sao Paulo (SP)', 'Salvador (BH)', 
                                                     'Brasilia (DF)', 'Rio de Janeiro (RJ)', 'Campo Grande (MS)', 
                                                     'Aracaju (SE)', 'Natal (RN)', 'Recife (PE)'])
    
    destination = st.selectbox('Select Destination City', ['Florianopolis (SC)', 'Sao Paulo (SP)', 'Salvador (BH)', 
                                                           'Brasilia (DF)', 'Rio de Janeiro (RJ)', 'Campo Grande (MS)', 
                                                           'Aracaju (SE)', 'Natal (RN)', 'Recife (PE)'])
    
    flight_class = st.selectbox('Select Flight Class', ['economic', 'firstClass', 'premium'])
    agency = st.selectbox('Select Agency', ['Rainbow', 'CloudFy', 'FlyingDrops'])

    week_no = st.number_input('Enter Week Number', min_value=1, max_value=52)
    week_day = st.number_input('Enter Week Day (1-7)', min_value=1, max_value=7)
    day = st.number_input('Enter Day (1-31)', min_value=1, max_value=31)

    # Creating the travel_dict based on user input
    boarding = 'from_' + boarding
    destination = 'destination_' + destination
    flight_class = 'flightType_' + flight_class
    agency = 'agency_' + agency

    boarding_city_list = ['from_Florianopolis (SC)', 'from_Sao_Paulo (SP)', 'from_Salvador (BH)', 
                          'from_Brasilia (DF)', 'from_Rio_de_Janeiro (RJ)', 'from_Campo_Grande (MS)', 
                          'from_Aracaju (SE)', 'from_Natal (RN)', 'from_Recife (PE)']
    destination_city_list = ['destination_Florianopolis (SC)', 'destination_Sao_Paulo (SP)', 
                             'destination_Salvador (BH)', 'destination_Brasilia (DF)', 
                             'destination_Rio_de_Janeiro (RJ)', 'destination_Campo_Grande (MS)', 
                             'destination_Aracaju (SE)', 'destination_Natal (RN)', 'destination_Recife (PE)']
    class_list = ['flightType_economic', 'flightType_firstClass', 'flightType_premium']
    agency_list = ['agency_Rainbow', 'agency_CloudFy', 'agency_FlyingDrops']

    travel_dict = dict()

    # Creating the one-hot encoded travel_dict
    for city in boarding_city_list:
        travel_dict[city] = 1 if city == boarding else 0
    for city in destination_city_list:
        travel_dict[city] = 1 if city == destination else 0
    for flight_class_type in class_list:
        travel_dict[flight_class_type] = 1 if flight_class_type == flight_class else 0
    for agency_type in agency_list:
        travel_dict[agency_type] = 1 if agency_type == agency else 0

    travel_dict['week_no'] = week_no
    travel_dict['week_day'] = week_day
    travel_dict['day'] = day

    # Perform prediction when the button is pressed
    if st.button('Predict Flight Price'):
        predicted_price = predict_price(travel_dict, rf_model, scaler_model)
        st.write(f"Predicted Flight Price Per Person: ${round(predicted_price, 2)}")

if __name__ == '__main__':
    main()
