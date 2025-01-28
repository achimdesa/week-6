# week-6
10 Academy Kifiya AI mastery training program week 6 challenge

# **Credit Risk Modeling and API Service**

This repository contains the full workflow of a credit risk modeling project, from data preprocessing and feature engineering to model training and deployment via a REST API using **FastAPI**. The project predicts whether a user transaction is likely to be fraudulent based on the user's transaction history.

## **Table of Contents**
- [Project Overview](#project-overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation Requirements](#installation-requirements)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Serving API](#model-serving-api)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)

## **Project Overview**

This project builds a credit risk prediction model that identifies whether a transaction is fraudulent or not, based on customer transaction data. The project follows these major steps:

1. **Task 1:** Data loading and cleaning.
2. **Task 2:** Exploratory Data Analysis (EDA)
3. **Task 3:** Feature Engineering
4. **Task 4:** Weight of Evidence (WoE) and Information Value (IV) transformation using `ScorecardPy`.
5. **Task 5:** Model training and hyperparameter tuning for Logistic Regression and Random Forest models.
6. **Task 6:** Serving the trained models using a REST API for real-time predictions.

## **Features**
- Data preprocessing including RFMS scoring, temporal feature extraction, and missing value handling.
- Weight of Evidence (WoE) binning and Information Value (IV) calculation.
- Model training using Logistic Regression and Random Forest.
- Hyperparameter tuning using GridSearchCV.
- REST API deployment using FastAPI for serving the trained models.
- API endpoints for making predictions on new transactions.

## **Directory Structure**

project-root/ │ ├── data/ │ ├── data.csv # Raw data file │ ├── xente_variable_definitions.csv # Data dictionary ├── models/ │ ├── logistic_regression_model.pkl # Trained logistic regression model │ ├── random_forest_model.pkl # Trained random forest model │ ├── woe_bins.pkl # WoE binning object │ ├── train_data.pkl # Training data used in model │ ├── test_data.pkl # Test data used in model ├── notebooks/ │ ├── task-1.ipynb # Task 1 notebook for data loading and cleaning │ ├── task-2.ipynb # Task 2 notebook for EDA and Feature Engineering │ ├── task-3.ipynb # Task 3 notebook for WoE/IV calculation │ ├── task-4.ipynb # Task 4 notebook for model training and tuning │ ├── task-5.ipynb # Task 5 notebook for serving model using API ├── scripts/ │ ├── load_data.py # Script to load data │ ├── feature_engineering.py # Feature extraction and RFMS score calculation │ ├── woe_iv_calculation.py # WoE and IV calculation script ├── api/ │ ├── main.py # FastAPI app for serving models ├── README.md # Project documentation ├── requirements.txt 


## **Installation Requirements**

To run this project locally, you need to have `Python 3.8+` installed. You can install the project dependencies using the following command:

```bash
pip install -r requirements.txt


The dependencies include:

pandas
numpy
scikit-learn
scorecardpy
FastAPI
joblib
uvicorn (for running FastAPI)
Data Preprocessing
The data used in this project is found in data/data.csv.
Preprocessing steps include:
Temporal feature extraction (e.g., hour, day, month).
Calculation of RFMS (Recency, Frequency, Monetary, Standard deviation) score.
Handling missing values and categorical encoding.
Weight of Evidence (WoE) and Information Value (IV) calculation using ScorecardPy.


### **Part 3: Model Training, API, and Contributing**


## **Model Training**

- The models are trained on features created during the preprocessing stage.
- Two models are trained:
  1. **Logistic Regression**
  2. **Random Forest**
- Both models are saved as `.pkl` files under the `models/` directory.
  

## **Model Serving API**

A REST API is created to serve the trained models and allow real-time predictions based on new transactions. The API is built using **FastAPI**.

## **Running the Application**

### Step 1: Clone the Repository

git clone https://github.com/achimdesa/week-6.git

Step 2: Navigate to the scripts/ directory:
cd scripts

Step 3: Run the FastAPI server using Uvicorn:
Run the below command:
uvicorn model_serving:app --reload

The API will be accessible at http://127.0.0.1:8000.

API Endpoints
1. POST /logistic-regression/
Description: Make a prediction using the Logistic Regression model.

Request Body:

json

{
  "ProductCategory": "airtime",
  "ChannelId": "3",
  "Amount": 1000,
  "Value": 1000,
  "transaction_hour": 12,
  "transaction_day": 15
}
Response:

json

{
  "prediction": "0", 
  "probability": 0.85
}

2. POST /random-forest/
Description: Make a prediction using the Random Forest model.

Request Body: Same as above.

Response:

json

{
  "prediction": "1", 
  "probability": 0.72
}
Contributing
If you'd like to contribute to this project, feel free to create a pull request or submit an issue.

