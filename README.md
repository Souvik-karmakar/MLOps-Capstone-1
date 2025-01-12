# **Travel and Tourism Analytics with Machine Learning and MLOps**  

## **Overview**  
This project explores the powerful intersection of **data analytics** and **machine learning** to revolutionize travel experiences. By leveraging datasets related to users, flights, and hotels, I’ve developed predictive and recommendation models that are scalable, interactive, and production-ready. The project also integrates **MLOps practices** to automate workflows, track models, and ensure seamless deployment.  

---

## **Features**  

### 🔍 **Flight Price Prediction**  
- Built a regression model to predict flight prices using the `flights.csv` dataset.  
- Optimized model performance through feature selection, training, and validation.

<img width="899" alt="flight_price_prediction_1" src="https://github.com/user-attachments/assets/e6af7db2-87ac-451e-924f-5354fa85a592" />
<img width="896" alt="Flight_price_prediction_2" src="https://github.com/user-attachments/assets/92f3dd4e-667c-4bac-b5ea-2b6a1cbfe98e" />
<img width="487" alt="flight_price_prediction_3" src="https://github.com/user-attachments/assets/5d384e1d-a533-419c-8772-21f204af4652" />

### 🌐 **REST API Development**  
- Developed a Flask-based API to serve flight price predictions in real time.  

### 📦 **Containerization with Docker**  
- Packaged the model and API for portability and ease of deployment.  

### 📈 **Kubernetes Deployment**  
- Ensured scalability and efficient load management by deploying the Dockerized application using Kubernetes.  

### 🔄 **Workflow Automation with Apache Airflow**  
- Designed and implemented DAGs to automate data preprocessing and model training workflows.  

### 🚀 **CI/CD Pipeline with Jenkins**  
- Automated the integration and deployment process using a Jenkins pipeline, ensuring reliable and consistent releases.  

### 🧬 **Model Tracking with MLFlow**  
- Managed model versions systematically and tracked performance metrics during iterations.  

### 👥 **Gender Classification Model**  
- Deployed a classification model to predict user gender based on the `users.csv` dataset.

<img width="928" alt="Gender Classification app image" src="https://github.com/user-attachments/assets/1600e998-497b-4f95-91ac-e422e2afe95c" />
<img width="956" alt="streamlit_run_app" src="https://github.com/user-attachments/assets/9692a33a-f79c-4d63-94c6-b3ba2101ff2a" />
<img width="722" alt="Gender_Classification_model_prediction" src="https://github.com/user-attachments/assets/593d9d65-8c2d-4d4f-a2fb-338008fe5d13" />

### 🏨 **Hotel Recommendation System**  
- Built a recommendation model for personalized hotel suggestions using user preferences and historical data.  
- Deployed an interactive **Streamlit app** for seamless data visualization and exploration.
  
<img width="958" alt="Hotel_recommendation_app" src="https://github.com/user-attachments/assets/2dc91fb3-8f78-415e-8309-df0323f00313" />



---

## **Tech Stack**  

### **Programming Languages & Libraries**  
- **Python**: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Streamlit, Flask  
- **MLOps Tools**: Docker, Kubernetes, Apache Airflow, Jenkins, MLFlow  

### **Platforms**  
- **Version Control**: GitHub  
- **Deployment**: Docker Hub, Kubernetes  

---

## **Project Structure**  

```plaintext  
├── data/  
│   ├── flights.csv  
│   ├── users.csv  
│   ├── hotels.csv  
├── models/  
│   ├── regression_model.py  
│   ├── gender_classification_model.py  
│   ├── recommendation_model.py  
├── api/  
│   ├── app.py (Flask API for flight price prediction)  
├── workflows/  
│   ├── airflow_dag.py (Airflow DAG for automation)  
├── deployment/  
│   ├── Dockerfile  
│   ├── kubernetes_deployment.yaml  
├── ci_cd/  
│   ├── Jenkinsfile  
├── app/  
│   ├── streamlit_app.py (Hotel recommendation interface)  
├── README.md  
└── requirements.txt  
```  

---

## **Setup Instructions**  

### **Prerequisites**  
- Python 3.8+  
- Docker and Kubernetes installed  
- Apache Airflow  
- Jenkins setup for CI/CD  

### **Steps to Run the Project**  

1. **Clone the Repository**  
   ```bash  
   git clone https://github.com/your-repo-url.git  
   cd your-repo-folder  
   ```  

2. **Install Dependencies**  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. **Run the Flask API**  
   ```bash  
   cd api  
   python app.py  
   ```  

4. **Dockerize the Application**  
   ```bash  
   docker build -t flight-price-predictor .  
   docker run -p 5000:5000 flight-price-predictor  
   ```  

5. **Deploy on Kubernetes**  
   ```bash  
   kubectl apply -f deployment/kubernetes_deployment.yaml  
   ```  

6. **Set Up Airflow DAGs**  
   - Follow the instructions in `workflows/airflow_dag.py`.  

7. **Access the Streamlit App**  
   ```bash  
   streamlit run app/streamlit_app.py  
   ```  

---

## **Evaluation Metrics**  

### **Regression Model**  
- RMSE (Root Mean Squared Error)  
- R² Score  

### **Classification Model**  
- Accuracy  
- Precision & Recall  

### **Recommendation System**  
- Precision@K  

---

## **Results & Insights**  
- **Flight Price Prediction** achieved a high R² score with minimal error.  
- **Gender Classification Model** provided accurate predictions with balanced precision and recall.  
- **Hotel Recommendation System** enhanced user satisfaction by offering tailored suggestions.  

---

## **GitHub Contributions**  
- Organized commits reflecting development progress.  
- Comprehensive documentation in the repository.  

---

## **Future Enhancements**  
- Incorporating more datasets for richer insights.  
- Expanding the recommendation engine for multi-modal preferences.  
- Enhancing CI/CD pipelines with more automation.  

---

## **Contact**  
For any questions or feedback, feel free to reach out:  
📧 Email: ksouvik98@gmil.com 
💼 LinkedIn: https://www.linkedin.com/in/souvik-karmakar83/ 


