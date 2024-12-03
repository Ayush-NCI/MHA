from flask import Flask, request, render_template
import joblib
import numpy as np
import boto3
import json

# AWS Lambda client
lambda_client = boto3.client('lambda', region_name='eu-west-1')


app = Flask(__name__)

# Load your pre-trained model
model = joblib.load('random_forest_model.pkl')

# Define encoding strategies for categorical variables
encoding_strategies = {
    'Gender_encoded': {'Male': 0, 'Female': 1, 'Others': 2},
    'self_employed': {'Yes': 1, 'No': 0},
    'family_history': {'Yes': 1, 'No': 0},
    'work_interfere': {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3},
    'no_employees': {'1-5': 0, '6-25': 1, '26-100': 2, '100-500': 3, '500-1000': 4, 'More than 1000': 5},
    'tech_company': {'Yes': 1, 'No': 0},
    'benefits': {'Yes': 1, 'No': 0, "Don't know": 2},
    'care_options': {'Yes': 1, 'No': 0, 'Not sure': 2},
    'wellness_program': {'Yes': 1, 'No': 0, "Don't know": 2},
    'seek_help': {'Yes': 1, 'No': 0, "Don't know": 2},
    'anonymity': {'Yes': 1, 'No': 0, "Don't know": 2},
    'leave': {'Very easy': 0, 'Somewhat easy': 1, 'Somewhat difficult': 3, "Don't know": 2},
    'mental_health_consequence': {'Yes': 1, 'No': 0, 'Maybe': 2},
    'phys_health_consequence': {'Yes': 1, 'No': 0, 'Maybe': 2},
    'coworkers': {'Yes': 1, 'No': 0, 'Some of them': 2},
    'supervisor': {'Yes': 1, 'No': 0, 'Some of them': 2},
    'mental_vs_physical': {'Yes': 1, 'No': 0, "Don't know": 2},
    'sentiment_encoded': {'Positive': 1, 'Negative': -1, 'Neutral': 0}
}


@app.route('/')
def index():
    # Define dynamic fields and their options
    dynamic_fields = {
        'benefits': ['Yes', 'No', "Don't know"],
        'care_options': ['Yes', 'No', 'Not sure'],
        'wellness_program': ['Yes', 'No', "Don't know"],
        'seek_help': ['Yes', 'No', "Don't know"],
        'anonymity': ['Yes', 'No', "Don't know"],
        'leave': ['Very easy', 'Somewhat easy', 'Somewhat difficult', "Don't know"],
        'mental_health_consequence': ['Yes', 'No', 'Maybe'],
        'phys_health_consequence': ['Yes', 'No', 'Maybe'],
        'coworkers': ['Yes', 'No', 'Some of them'],
        'supervisor': ['Yes', 'No', 'Some of them'],
        'mental_vs_physical': ['Yes', 'No', "Don't know"]
    }
    return render_template('index.html', dynamic_fields=dynamic_fields, prediction=None)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()

    # Process data in the exact order used during training
    feature_order = [
        'Age', 'Gender_encoded', 'self_employed', 'family_history', 'work_interfere',
        'no_employees', 'tech_company', 'benefits', 'care_options', 'wellness_program',
        'seek_help', 'anonymity', 'leave', 'mental_health_consequence',
        'phys_health_consequence', 'coworkers', 'supervisor', 'mental_vs_physical', 'sentiment_encoded'
    ]
    
    processed_data = [
        float(data[field]) if field == 'Age' else encoding_strategies[field][data[field]]
        for field in feature_order
    ]

    # Check length
    if len(processed_data) != len(feature_order):
        return render_template('index.html', dynamic_fields=dynamic_fields, prediction="Feature count mismatch. Please check input data.")

    # Convert to numpy array and predict
    processed_data = np.array([processed_data])
    prediction = model.predict(processed_data)
    result = "Needs Treatment" if prediction[0] == 1 else "No Treatment Needed"

     # Prepare data for Lambda
    lambda_payload = {
        "prediction_result": result,
        "user_data": data
    }

    try:
        # Invoke Lambda function
        response = lambda_client.invoke(
            FunctionName='x23178248_save_prediction_result',
            InvocationType='Event',  # Use 'Event' for asynchronous invocation
            Payload=json.dumps(lambda_payload)
        )
        print(f"Lambda invocation response: {response}")
    except Exception as e:
        print(f"Error invoking Lambda: {e}")


    # Render the prediction on the same page
    dynamic_fields = {
        'benefits': ['Yes', 'No', "Don't know"],
        'care_options': ['Yes', 'No', 'Not sure'],
        'wellness_program': ['Yes', 'No', "Don't know"],
        'seek_help': ['Yes', 'No', "Don't know"],
        'anonymity': ['Yes', 'No', "Don't know"],
        'leave': ['Very easy', 'Somewhat easy', 'Somewhat difficult',  "Don't know"],
        'mental_health_consequence': ['Yes', 'No', 'Maybe'],
        'phys_health_consequence': ['Yes', 'No', 'Maybe'],
        'coworkers': ['Yes', 'No', 'Some of them'],
        'supervisor': ['Yes', 'No', 'Some of them'],
        'mental_vs_physical': ['Yes', 'No', "Don't know"]
    }
    return render_template('index.html', dynamic_fields=dynamic_fields, prediction=result)


if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(host='0.0.0.0', port=8080, debug=True)
