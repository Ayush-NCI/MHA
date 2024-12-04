from flask import Flask, request, render_template
import joblib
import numpy as np
import boto3
import json
from sklearn.preprocessing import StandardScaler
import boto3
from datetime import datetime
from decimal import Decimal 

# AWS Lambda client
lambda_client = boto3.client('lambda', region_name='eu-west-1')


app = Flask(__name__)

# Load your pre-trained model
model = joblib.load('rfmodel.pkl')

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
    'physical_health_consequence': {'Yes': 1, 'No': 0, 'Maybe': 2},
    'coworkers': {'Yes': 1, 'No': 0, 'Some of them': 2},
    'supervisor': {'Yes': 1, 'No': 0, 'Some of them': 2},
    'mental_vs_physical': {'Yes': 1, 'No': 0, "Don't know": 2},
    'sentiment_encoded': {'Positive': 1, 'Negative': -1, 'Neutral': 0}
}



# Function to save user data into DynamoDB
def save_to_dynamodb(data, prediction_result):
    try:
        # Create a DynamoDB client
        dynamodb = boto3.resource('dynamodb', region_name='eu-west-2')  # Change region if needed
        table = dynamodb.Table('x23178248_user_predictions')  # Your DynamoDB table name

        # Create a unique user_id (e.g., using timestamp or UUID)
        user_id = str(datetime.now().timestamp())  # You can use UUID for uniqueness
        # Get the current timestamp (or use the timestamp field if needed)
        timestamp = Decimal(datetime.now().timestamp()) 
        print(timestamp)

        # Prepare the data to be saved
        user_data = {
            'user_id': user_id,
            'timestamp': timestamp,
            'Age': Decimal(data.get('Age', 0)),  # Convert 'Age' to Decimal
            'Gender_encoded': data.get('Gender_encoded', None),  # Map 'Gender' to 'Gender_encoded'
            'self_employed': encoding_strategies['self_employed'].get(data.get('self_employed', ''), None),
            'family_history': encoding_strategies['family_history'].get(data.get('family_history', ''), None),
            'work_interfere': encoding_strategies['work_interfere'].get(data.get('work_interfere', ''), None),
            'no_employees': encoding_strategies['no_employees'].get(data.get('no_employees', ''), None),
            'tech_company': encoding_strategies['tech_company'].get(data.get('tech_company', ''), None),
            'benefits': encoding_strategies['benefits'].get(data.get('benefits', ''), None),
            'care_options': encoding_strategies['care_options'].get(data.get('care_options', ''), None),
            'wellness_program': encoding_strategies['wellness_program'].get(data.get('wellness_program', ''), None),
            'seek_help': encoding_strategies['seek_help'].get(data.get('seek_help', ''), None),
            'anonymity': encoding_strategies['anonymity'].get(data.get('anonymity', ''), None),
            'leave': encoding_strategies['leave'].get(data.get('leave', ''), None),
            'mental_health_consequence': encoding_strategies['mental_health_consequence'].get(data.get('mental_health_consequence', ''), None),
            'physical_health_consequence': encoding_strategies['physical_health_consequence'].get(data.get('phys_health_consequence', ''), None),
            'coworkers': encoding_strategies['coworkers'].get(data.get('coworkers', ''), None),
            'supervisor': encoding_strategies['supervisor'].get(data.get('supervisor', ''), None),
            'mental_vs_physical': encoding_strategies['mental_vs_physical'].get(data.get('mental_vs_physical', ''), None),
            'sentiment_encoded': encoding_strategies['sentiment_encoded'].get(data.get('sentiment_encoded', ''), None),
            'prediction_result': prediction_result
        }

        # Save the data to DynamoDB
        table.put_item(Item=user_data)
        print("Data saved to DynamoDB successfully.")

    except Exception as e:
        # Log the error but allow prediction to proceed
        print(f"Error saving data to DynamoDB: {e}")

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
        'physical_health_consequence': ['Yes', 'No', 'Maybe'],
        'coworkers': ['Yes', 'No', 'Some of them'],
        'supervisor': ['Yes', 'No', 'Some of them'],
        'mental_vs_physical': ['Yes', 'No', "Don't know"]
    }
    return render_template('index.html', dynamic_fields=dynamic_fields, prediction=None)

def fetch_data_from_dynamodb():
    """
    Fetch all data from the DynamoDB table as-is.
    """
    try:
        # Create a DynamoDB client
        dynamodb = boto3.resource('dynamodb', region_name='eu-west-2')  # Change region if needed
        table = dynamodb.Table('x23178248_user_predictions')  # Replace with your table name

        # Scan the table to fetch all items
        response = table.scan()
        items = response.get('Items', [])

        return items
    except Exception as e:
        print(f"Error fetching data from DynamoDB: {e}")
        return []


@app.route('/view_data')
def view_data():
    # Fetch raw data from DynamoDB
    dynamo_data = fetch_data_from_dynamodb()

    # Analyze data for charts
    prediction_counts = {'Needs Treatment': 0, 'No Treatment Needed': 0}
    age_distribution = []
    gender_distribution = {'Male': 0, 'Female': 0, 'Others': 0}
    work_interfere_levels = {'Never': 0, 'Rarely': 0, 'Sometimes': 0, 'Often': 0}
    benefits_vs_prediction = {'Yes': {'Needs Treatment': 0, 'No Treatment Needed': 0},
                              'No': {'Needs Treatment': 0, 'No Treatment Needed': 0},
                              "Don't know": {'Needs Treatment': 0, 'No Treatment Needed': 0}}

    # Decoding and analysis
    for item in dynamo_data:
        # Decode Gender
        gender_code = item.get('Gender_encoded', 'Others')  # Default to 'Others' if not found
    
        # Update the gender distribution count based on the gender_code value
        if gender_code in gender_distribution:
            gender_distribution[gender_code] += 1
        else:
            # If there is an unexpected value, classify it as 'Others'
            gender_distribution['Others'] += 1
            
        

        # Decode and categorize work interfere
        work_interfere_code = item.get('work_interfere', None)
        work_interfere = next((key for key, value in encoding_strategies['work_interfere'].items() if value == work_interfere_code), 'Unknown')
        if work_interfere in work_interfere_levels:
            work_interfere_levels[work_interfere] += 1

        # Decode benefits and categorize with predictions
        benefits_code = item.get('benefits', None)
        benefits = next((key for key, value in encoding_strategies['benefits'].items() if value == benefits_code), "Unknown")
        prediction = item.get('prediction_result', 'No Treatment Needed')
        if benefits in benefits_vs_prediction:
            benefits_vs_prediction[benefits][prediction] += 1

        # Predictions
        prediction_counts[prediction] += 1

        # Age distribution
        try:
            age_distribution.append(int(item.get('Age', 0)))
        except ValueError:
            pass  # Skip invalid ages

    return render_template(
        'view_data.html',
        data=dynamo_data,
        keys=dynamo_data[0].keys() if dynamo_data else [],
        prediction_counts=prediction_counts,
        age_distribution=age_distribution,
        gender_distribution=gender_distribution,
        work_interfere_levels=work_interfere_levels,
        benefits_vs_prediction=benefits_vs_prediction
    )



@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()

    # Process data in the exact order used during training
    feature_order = [
        'Age', 'Gender_encoded', 'self_employed', 'family_history', 'work_interfere',
        'no_employees', 'tech_company', 'benefits', 'care_options', 'wellness_program',
        'seek_help', 'anonymity', 'leave', 'mental_health_consequence',
        'physical_health_consequence', 'coworkers', 'supervisor', 'mental_vs_physical', 'sentiment_encoded'
    ]
    print("data coming from form-----------------------")
    print(data)
    processed_data = [
        float(data[field]) if field == 'Age' else encoding_strategies[field][data[field]]
        for field in feature_order
    ]

    # Check length
    if len(processed_data) != len(feature_order):
        return render_template('index.html', dynamic_fields=dynamic_fields, prediction="Feature count mismatch. Please check input data.")

    
    processed_data = np.array([processed_data])  # Ensure that processed_data is a 2D array
    print("before preprocessing----------------------")
    print(processed_data)
    

    age_index = feature_order.index('Age')  # Get the index of the 'Age' column in feature_order

    # Scale only the 'Age' column
    scaler = StandardScaler()

    # Apply scaling to 'Age' column
    processed_data[:, age_index] = scaler.fit_transform(processed_data[:, age_index].reshape(-1, 1)).flatten()
    
    print("after preprocessing----------------------")
    print(processed_data)
    prediction = model.predict(processed_data)
    print(prediction)
    result = "Needs Treatment" if prediction[0] == 1 else "No Treatment Needed"


    # Save the response and prediction result to DynamoDB
    save_to_dynamodb(data, result)

    # Render the prediction on the same page
    dynamic_fields = {
        'benefits': ['Yes', 'No', "Don't know"],
        'care_options': ['Yes', 'No', 'Not sure'],
        'wellness_program': ['Yes', 'No', "Don't know"],
        'seek_help': ['Yes', 'No', "Don't know"],
        'anonymity': ['Yes', 'No', "Don't know"],
        'leave': ['Very easy', 'Somewhat easy', 'Somewhat difficult',  "Don't know"],
        'mental_health_consequence': ['Yes', 'No', 'Maybe'],
        'physical_health_consequence': ['Yes', 'No', 'Maybe'],
        'coworkers': ['Yes', 'No', 'Some of them'],
        'supervisor': ['Yes', 'No', 'Some of them'],
        'mental_vs_physical': ['Yes', 'No', "Don't know"]
    }
    return render_template('index.html', dynamic_fields=dynamic_fields, prediction=result)


if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(host='0.0.0.0', port=8080, debug=True)
