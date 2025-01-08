# Use the official Python image as a parent image
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy requirements.txt separately and install dependencies first
COPY requirements.txt ./requirements.txt

# Copy the machine learning model file into the container
COPY ./model/tuned_logistic_regression_model_1.pkl /app/model/tuned_logistic_regression_model_1.pkl
COPY ./model/pca_1.pkl /app/model/pca_1.pkl
COPY ./model/scaler_1.pkl /app/model/scaler_1.pkl

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# Expose port 8000 for the Flask app to listen on
EXPOSE 8000

# Define environment variable for Flask
ENV FLASK_APP=app.py

# Run the Flask application
# CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]
CMD ["python", "app.py"]