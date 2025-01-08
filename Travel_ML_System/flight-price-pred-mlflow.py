#Import Libraries:
import mlflow
import pandas as pd
import numpy as np
import logging

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


#Configure Logging:
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#Set MLflow Tracking URI:
mlflow.set_tracking_uri("http://127.0.0.1:8000")

#Start an MLflow Run:
mlflow.start_run()
#mlflow.create_experiment("Flight_package_prediction")
#Load Data:
df = pd.read_csv("flights.csv")

# Change travel date into a datetime object
df['date'] = pd.to_datetime(df['date'])
        
# Extracting WeekNo., Month, Year, Weekday from date column
df['week_day'] = df['date'].dt.weekday
df['month'] = df['date'].dt.month
df['week_no'] = df['date'].dt.isocalendar().week
df['year'] = df['date'].dt.year
df['day'] = df['date'].dt.day
       
# Renaming the Column name
df.rename(columns={"to":"destination"},inplace=True)
        
# Creating a new feature using distance and time columns
df['flight_speed']=round(df['distance']/df['time'],2)
        
# Example of one-hot encoding
df = pd.get_dummies(df, columns=['from','destination','flightType','agency'])

        
#Dropping irrelavent features
df.drop(columns=['time','flight_speed','month','year','distance'],axis=1,inplace=True)
        
#Separate features (X) and target variable (Y)
X = df.drop('price', axis=1)  # Features
Y = df['price']               # Target variable
               
        
#Renaming the coulmns 
X.rename(columns={'from_Sao Paulo (SP)':'from_Sao_Paulo (SP)','from_Rio de Janeiro (RJ)':'from_Rio_de_Janeiro (RJ)','from_Campo Grande (MS)':'from_Campo_Grande (MS)',
                                  'destination_Sao Paulo (SP)':'destination_Sao_Paulo (SP)','destination_Rio de Janeiro (RJ)':'destination_Rio_de_Janeiro (RJ)','destination_Campo Grande (MS)':'destination_Campo_Grande (MS)'},inplace=True)

#Sorting the features based on our output requirments
features_ordering=['from_Florianopolis (SC)','from_Sao_Paulo (SP)','from_Salvador (BH)','from_Brasilia (DF)','from_Rio_de_Janeiro (RJ)','from_Campo_Grande (MS)','from_Aracaju (SE)',
 'from_Natal (RN)','from_Recife (PE)','destination_Florianopolis (SC)','destination_Sao_Paulo (SP)','destination_Salvador (BH)','destination_Brasilia (DF)','destination_Rio_de_Janeiro (RJ)',
 'destination_Campo_Grande (MS)','destination_Aracaju (SE)','destination_Natal (RN)','destination_Recife (PE)','flightType_economic','flightType_firstClass','flightType_premium',
 'agency_Rainbow','agency_CloudFy','agency_FlyingDrops','week_no','week_day','day']
        
#Ordering features based on flask output
X= X[features_ordering]

#Split Data into Train and Test Sets:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

#Standardize Data:
scaler_new = StandardScaler()
X_train = scaler_new.fit_transform(X_train)
X_test = scaler_new.transform(X_test)

#Hyperparameter tuning and cross validation using GridsearchCV      
param_dict = {
            'n_estimators': [300],
            'max_depth': [15],
            'min_samples_split': [10],
            'max_features': ['sqrt',27],
            'n_jobs': [2]
        }
rf_model = RandomForestRegressor(random_state=42)
rf_grid = GridSearchCV(estimator=rf_model,
                                     param_grid=param_dict,
                                     cv=3, verbose=2, scoring='r2')

rf_grid.fit(X_train, Y_train) 
            
rf_optimal_model = rf_grid.best_estimator_

Y_train_pred = rf_optimal_model.predict(X_train)
Y_test_pred = rf_optimal_model.predict(X_test)

actual=Y_test
predicted=Y_test_pred

#Evaluation Metrics
MSE = mean_squared_error(actual, predicted)
MAE = mean_absolute_error(actual, predicted)
RMSE = np.sqrt(MSE)
R2 = r2_score(actual, predicted) 

#Log Parameters and Metrics to MLflow:
mlflow.log_param("test_size", 0.3)
mlflow.log_param("random_state", 42)
mlflow.log_param("n_estimators", 300)
mlflow.log_param("max_depth", 15)
mlflow.log_metric("MAE", MAE)
mlflow.log_metric("MSE", MSE)
mlflow.log_metric("RMSE", RMSE)
mlflow.log_metric("R2", R2)

#Log the Trained Model to MLflow:
mlflow.sklearn.log_model(rf_optimal_model, "random_forest_model")

#Register the Model Version:
#mlflow.register_model("runs:/<RUN_ID>/random_forest_model", "FlightPackagePriceModel")

#End the MLflow Run:
mlflow.end_run()