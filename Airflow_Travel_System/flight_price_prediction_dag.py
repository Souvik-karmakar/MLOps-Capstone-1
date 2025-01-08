from datetime import datetime
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from utils.data_ingestion import DataLoader
from utils.data_transformation import DataTransformer
from utils.model_training import RandomForestModel


# Define your file paths here
data_file_path = './plugins/flights.csv'

# Create instances of your classes
data_loader = DataLoader(data_file_path)
data_transformer = DataTransformer(data_loader.load_data())
X, Y = data_transformer.transform()
random_forest_model = RandomForestModel(X, Y)
#random_forest_model = RandomForestModel(data_transformer.transform())


# Define the default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Create the DAG
dag = DAG (
    dag_id = 'flight_price_prediction_dag',
    default_args=default_args,
    description='A DAG for flight price prediction',
    catchup=False,
    schedule_interval= '@once')  # Define your desired schedule interval

# Task to load data
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=data_loader.load_data,
    dag=dag
)

# Task to transform data
transform_data_task = PythonOperator(
    task_id='transform_data_task',
    python_callable=data_transformer.transform,
    dag=dag
)

# Task to run random forest model
random_forest_task = PythonOperator(
    task_id='random_forest_task',
    python_callable=random_forest_model.random_forest,
    dag=dag
)

# Define the task dependencies
load_data_task >> transform_data_task >> random_forest_task