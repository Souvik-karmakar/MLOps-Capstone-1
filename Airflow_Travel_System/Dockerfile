FROM apache/airflow:latest

USER root

# Install necessary dependencies like git
RUN apt-get update && \
    apt-get -y install git && \
    apt-get clean

# Revert to the airflow user for safety and to maintain security context
USER airflow

# Make sure airflow is installed (this might be redundant if using the official image)
RUN pip install --upgrade pip && \
    pip install apache-airflow
