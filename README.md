# 🛡️ Insurance Fraud Detection System

A Machine Learning-based web application that predicts whether an insurance claim is **Fraudulent** or **Non-Fraudulent** using a trained **Decision Tree Classifier**.

The project demonstrates an end-to-end Machine Learning workflow, from data preprocessing and model training to building a Flask REST API and deploying the application on Vercel.

---

## 🚀 Live Demo

🌐 **Live Application:**  
https://fraudclaimdetection.vercel.app

### API Health Check

https://fraudclaimdetection.vercel.app/api/health

The health endpoint verifies that the model, dataset, and label encoders are successfully loaded.

---

## 📌 Features

- 🔍 Predicts whether an insurance claim is **Fraudulent** or **Non-Fraudulent**
- 🤖 Uses a trained **Decision Tree Classifier**
- 🌐 Modern web interface using HTML, CSS, and JavaScript
- ⚡ Flask REST API for Machine Learning predictions
- 🎲 Random sample claim generator
- ✏️ Editable insurance claim details
- 📊 Prediction results with model information
- 🧠 Decision-tree based explanation of prediction reasons
- 📈 Model performance information
- ☁️ Deployed on Vercel

---

## 🛠️ Tech Stack

### Frontend

- HTML5
- CSS3
- JavaScript

### Backend

- Python
- Flask
- Flask-CORS

### Machine Learning

- Pandas
- NumPy
- Scikit-learn
- Joblib

### Deployment

- Vercel

---

## 📊 Machine Learning Workflow

The project follows an end-to-end Machine Learning workflow:

1. Data Collection
2. Data Cleaning
3. Missing Value Handling
4. Feature Engineering
5. Categorical Feature Encoding
6. Class Balancing using SMOTE
7. Model Training
8. Model Evaluation
9. Decision Tree Selection
10. Model Serialization using Joblib
11. Flask REST API Development
12. Web Frontend Integration
13. Deployment on Vercel

---

## 🤖 Models Evaluated

| Model | Accuracy |
|---|---:|
| Logistic Regression | 73% |
| Decision Tree | **80%** ✅ |
| Random Forest | 79% |

### Selected Model

**Decision Tree Classifier**

The Decision Tree model was selected based on its performance in the project, achieving an accuracy of approximately **80%**.

---

## 🔌 API Endpoints

The application provides the following REST API endpoints:

### 1. Health Check

```http
GET /api/health
