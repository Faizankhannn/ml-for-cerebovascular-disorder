from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('stroke_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

# Features used for the model
FEATURES = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'gender_Male', 'ever_married_Yes',
    'work_type_Self-employed', 'work_type_Private', 'work_type_Govt_job', 'work_type_children',
    'Residence_type_Urban',
    'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes', 'smoking_status_Unknown'
]

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = request.form
    try:
        # Map form data to the required features
        user_data = {
            'age': float(form_data['age']),
            'hypertension': int(form_data['hypertension']),
            'heart_disease': int(form_data['heart_disease']),
            'avg_glucose_level': float(form_data['avg_glucose_level']),
            'bmi': float(form_data['bmi']),
            'gender_Male': 1 if form_data['gender'] == 'Male' else 0,
            'ever_married_Yes': 1 if form_data['ever_married'] == 'Yes' else 0,
            'work_type_Self-employed': 1 if form_data['work_type'] == 'Self-employed' else 0,
            'work_type_Private': 1 if form_data['work_type'] == 'Private' else 0,
            'work_type_Govt_job': 1 if form_data['work_type'] == 'Govt_job' else 0,
            'work_type_children': 1 if form_data['work_type'] == 'Children' else 0,
            'Residence_type_Urban': 1 if form_data['residence_type'] == 'Urban' else 0,
            'smoking_status_formerly smoked': 1 if form_data['smoking_status'] == 'formerly smoked' else 0,
            'smoking_status_never smoked': 1 if form_data['smoking_status'] == 'never smoked' else 0,
            'smoking_status_smokes': 1 if form_data['smoking_status'] == 'smokes' else 0,
            'smoking_status_Unknown': 1 if form_data['smoking_status'] == 'unknown' else 0
        }

        # Convert to DataFrame and preprocess
        input_df = pd.DataFrame([user_data])
        for col in FEATURES:
            if col not in input_df.columns:
                input_df[col] = 0
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0, 1]

        # Result message
        if prediction == 1:
            result = f"High Risk of Stroke (Confidence: {prediction_proba:.2f})"
        else:
            result = f"Low Risk of Stroke (Confidence: {1 - prediction_proba:.2f})"

        return render_template('result.html', result=result)

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
