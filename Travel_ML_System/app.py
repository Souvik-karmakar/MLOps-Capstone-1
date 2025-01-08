#Import required libraries
import pandas as pd
import pickle

#Import flask library
from flask import Flask, render_template, request, jsonify

#function to predict the price
def predict_price(input_data, model, scaler):
    
    # Prepare the input data

    # Initialize an empty DataFrame
    df_input2 = pd.DataFrame([input_data])

    # Independent features 
    X = df_input2 

    # Scale the data using the same scaler used during training Standard Scaler
    X = scaler.transform(X)

    # Make predictions using the trained Decision model
    y_prediction = model.predict(X)

    return y_prediction[0]


app = Flask(__name__)


@app.route('/',
           methods=['GET', 'POST'])
def predict():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def index():
    if request.method == 'POST':
        # Get input data from the form
        boarding = str(request.form.get('from'))
        destination = str(request.form.get('Destination'))
        selected_flight_class = str(request.form.get('flightType'))
        selected_agency = str(request.form.get('agency'))
        week_no = int(request.form.get('week_no'))
        week_day = int(request.form.get('week_day'))
        day = int(request.form.get('day'))

        boarding = 'from_' + boarding
        boarding_city_list = ['from_Florianopolis (SC)',
                              'from_Sao_Paulo (SP)',
                              'from_Salvador (BH)',
                              'from_Brasilia (DF)',
                              'from_Rio_de_Janeiro (RJ)',
                              'from_Campo_Grande (MS)',
                              'from_Aracaju (SE)',
                              'from_Natal (RN)',
                              'from_Recife (PE)']

        destination = 'destination_' + destination
        destination_city_list = ['destination_Florianopolis (SC)',
                                 'destination_Sao_Paulo (SP)',
                                 'destination_Salvador (BH)',
                                 'destination_Brasilia (DF)',
                                 'destination_Rio_de_Janeiro (RJ)',
                                 'destination_Campo_Grande (MS)',
                                 'destination_Aracaju (SE)',
                                 'destination_Natal (RN)',
                                 'destination_Recife (PE)']

        selected_flight_class = 'flightType_' + selected_flight_class
        class_list = ['flightType_economic',
                      'flightType_firstClass',
                      'flightType_premium']

        selected_agency = 'agency_' + selected_agency
        agency_list = ['agency_Rainbow',
                       'agency_CloudFy',
                       'agency_FlyingDrops']

        travel_dict = dict()

        for city in boarding_city_list:
            if city[:-5] != boarding:
                travel_dict[city] = 0
            else:
                travel_dict[city] = 1
        for city in destination_city_list:
            if city[:-5] != destination:
                travel_dict[city] = 0
            else:
                travel_dict[city] = 1
        for flight_class in class_list:
            if flight_class != selected_flight_class:
                travel_dict[flight_class] = 0
            else:
                travel_dict[selected_flight_class] = 1
        for agency in agency_list:
            if agency != selected_agency:
                travel_dict[agency] = 0
            else:
                travel_dict[selected_agency] = 1
        travel_dict['week_no'] = week_no
        travel_dict['week_day'] = week_day
        travel_dict['day'] = day

        #scaler_model = pickle.load(open("./model/scaling.pkl", 'rb'))
        scaler_model = pickle.load(open("model/scaling_1.pkl", 'rb'))
        #rf_model = pickle.load(open("./model/rf_model.pkl", 'rb'))
        rf_model = pickle.load(open("model/rf_model_1.pkl", 'rb'))
        # Perform prediction using the custom_input dictionary
        predicted_price = str(round(predict_price(travel_dict, rf_model, scaler_model), 2))
        # print(f'Predicted Flight Price Per Person: ${predicted_price}')

        return jsonify({'prediction': predicted_price})


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", port=8000)