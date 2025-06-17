# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib  # For saving and loading models

# Load the dataset
data = pd.read_csv("dataset.csv")

# Handle missing values
data['bmi'].fillna(data['bmi'].median(), inplace=True)
data['smoking_status'].fillna('Unknown', inplace=True)

# One-hot encode categorical variables
categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Drop irrelevant columns
data.drop(columns=['id'], inplace=True)

# Separate features (X) and target (y)
X = data.drop(columns=['stroke'])
y = data['stroke']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Train a Random Forest classifier
model = RandomForestClassifier(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Save the model and scaler for future use
joblib.dump(model, 'stroke_prediction_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Make predictions
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_pred_prob))

# Function to preprocess new inputs and classify
def classify_new_input(input_data):
    """
    Classifies new input using the trained model.
    Args:
    - input_data (dict): A dictionary containing input features.
    
    Returns:
    - int: Predicted class (0 or 1).
    """
    # Convert the input dictionary to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Handle missing columns (due to one-hot encoding)
    for col in scaler.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Standardize the input
    input_scaled = scaler.transform(input_df)
    
    # Predict using the trained model
    prediction = model.predict(input_scaled)
    return prediction[0]

# Example usage
new_input = {
    'age': 50,
    'hypertension': 1,
    'heart_disease': 0,
    'avg_glucose_level': 105.5,
    'bmi': 28.7,
    'gender_Male': 1,
    'ever_married_Yes': 1,
    'work_type_Self-employed': 0,
    'work_type_Private': 1,
    'work_type_Govt_job': 0,
    'work_type_children': 0,
    'Residence_type_Urban': 1,
    'smoking_status_formerly smoked': 0,
    'smoking_status_never smoked': 1,
    'smoking_status_smokes': 0,
    'smoking_status_Unknown': 0
}

print("\nPredicted Class for New Input:", classify_new_input(new_input))