# Use official Python Image as Parent Image
FROM python:3.9-slim-buster

# Set working directory inside the container
WORKDIR /app

# Copy only requirements.txt first to leverage caching
COPY requirements.txt ./requirements.txt

# Install all dependencies and packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and other necessary files
COPY . /app

# Copy machine learning models into the container
COPY ./model/scaling.pkl /app/model/scaling.pkl
COPY ./model/rf_model.pkl /app/model/rf_model.pkl

# Expose the port 8000 for the Flask app to listen on
EXPOSE 8000

# Define environment variable for Flask
ENV FLASK_APP=app.py

# Use a production-ready WSGI server (e.g., Gunicorn)
#CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]
CMD ["python", "app.py"]


