from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("train.csv")

# Preprocessing
# Fill empty values in 'Credit_Classification' with 'Standard'
df['Credit_Classification'] = df['Credit_Classification'].fillna('Standard')

# Convert 'Credit_Classification' to binary classification
df['Credit_Classification'] = df['Credit_Classification'].map({'Good': 1, 'Standard': 0, 'Poor': 0})

# Remove string attributes
df = df[['Delay_from_due_date', 'Num_of_Delayed_Payment', 'Monthly_Balance', 'Amount_invested_monthly', 'Total_EMI_per_month', 'Credit_Classification']]

# Fill empty values with the mean
df.fillna(df.mean(), inplace=True)

# Normalization
scaler = MinMaxScaler()
df[['Delay_from_due_date', 'Num_of_Delayed_Payment', 'Monthly_Balance', 'Amount_invested_monthly', 'Total_EMI_per_month']] = scaler.fit_transform(df[['Delay_from_due_date', 'Num_of_Delayed_Payment', 'Monthly_Balance', 'Amount_invested_monthly', 'Total_EMI_per_month']])

# Splitting the dataset
X = df.drop('Credit_Classification', axis=1)
y = df['Credit_Classification']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'decision_tree_model.joblib')

# Prediction function
def predict_credit_score(Delay_from_due_date, Num_of_Delayed_Payment, Monthly_Balance, Amount_invested_monthly, Total_EMI_per_month):
    # Load the model
    model = joblib.load('decision_tree_model.joblib')

    # Preprocess the input
    input_data = [[Delay_from_due_date, Num_of_Delayed_Payment, Monthly_Balance, Amount_invested_monthly, Total_EMI_per_month]]

    # Normalize the input
    input_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        result = "Good"
    else:
        result = "Bad"

    return result

# Performance Measurement
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(y_true, y_pred))
    plt.title(all_sample_title, size=15)
    plt.show()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    Delay_from_due_date = float(request.form['Delay_from_due_date'])
    Num_of_Delayed_Payment = float(request.form['Num_of_Delayed_Payment'])
    Monthly_Balance = float(request.form['Monthly_Balance'])
    Amount_invested_monthly = float(request.form['Amount_invested_monthly'])
    Total_EMI_per_month = float(request.form['Total_EMI_per_month'])

    # Perform Prediction
    result = predict_credit_score(Delay_from_due_date, Num_of_Delayed_Payment, Monthly_Balance, Amount_invested_monthly, Total_EMI_per_month)

    # Performance Measure
    y_pred = model.predict(X_test)

    plot_confusion_matrix(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    return render_template('home.html', prediction_text=f'Customer credit score: {result}, F1 Score: {f1}')

if __name__ == "__main__":
    app.run(debug=True)

