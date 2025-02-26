# Customer Segmentation & AI Email Generator

## Overview
This project is a Flask-based web application that predicts customer segments based on their shopping behavior and generates personalized marketing emails accordingly. The model categorizes users into different customer types and provides targeted email templates.

## Features
- Accepts user inputs related to shopping behavior.
- Uses a pre-trained Random Forest model for customer segmentation.
- Generates AI-powered personalized email templates.
- Simple and interactive UI using HTML, CSS, and Flask.

## Technologies Used
- **Frontend**: HTML, CSS
- **Backend**: Python, Flask
- **Machine Learning Model**: Random Forest (Trained using scikit-learn)
- **Storage**: Joblib (for model persistence)

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/kv-06/Google-Hackathon.git
cd Google-Hackathon
```
### 2. Install Dependencies
Run the following command to install all required libraries:

```bash
pip install flask joblib scikit-learn transformers torch pandas datasets
```
### 3. Ensure Model File is in Place
Ensure that the random_forest_customer_model.joblib file is present inside the model/ directory.

### 4. Run the Flask Application
```bash
python app.py
```
The application will be accessible at: http://127.0.0.1:5000/

## Usage
1. Open the web application in a browser.
2. Enter customer behavior details such as days since last visit, total purchases, etc.
3. Click "Predict & Generate Email".
4. The system will classify the customer and generate a personalized email template.


## Customer Segmentation Categories
The model categorizes customers into:
- Cart Abandoner:  Users who added items to the cart but didn’t purchase.
- Frequent Buyer: Regular shoppers.
- Inactive Customer: Users who haven’t visited in a while.
- New User: Recently joined customers.

## Future Enhancements
Currently I have kept templates for the emails, a fine tuned hugging face gpt2 model will be used in the actual implementation to generate the emails 
