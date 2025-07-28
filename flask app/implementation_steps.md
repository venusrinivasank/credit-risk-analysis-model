# 🧱 Step-by-Step Implementation Guide

## 1️⃣ Install Required Libraries

- Ensure you have the necessary Python packages installed:

- command: pip install flask pandas scikit-learn matplotlib seaborn joblib
- run this in terminal of the folder you're developing the project

## 2️⃣ Prepare Your Dataset

- Load the dataset from train.csv.
- download the dataset from the link provided and move to the folder where you're creating your project

- Fill missing values in the Credit_Classification column with "Standard".

- Map Credit_Classification into binary format:

  - Good → 1

  - Standard/Poor → 0

- Drop irrelevant string attributes, keeping only numeric predictors and the target label.

## 3️⃣ Clean & Normalize Data

- Fill missing numeric values with column means.

- Normalize selected numeric columns using MinMaxScaler for better model performance.

## 4️⃣ Train Your Decision Tree Model

- Split the data into training and test sets (80:20 ratio).

- Train a DecisionTreeClassifier using scikit-learn.

- Save the trained model as decision_tree_model.joblib for later use.

## 5️⃣ Build the Flask Web Application

- Create app.py to define routes and serve predictions:

  - /: Displays the input form.

  -/predict: Takes form input, predicts credit rating, and displays result along with F1 Score and confusion matrix.

## 6️⃣ Create the HTML User Interface

- Design home.html under the /templates directory.

- Include input fields for:

  - Delay from Due Date

  - Number of Delayed Payments

  - Monthly Balance

  - Amount Invested Monthly

  - Total EMI per Month

- Display prediction result after form submission.

## 7️⃣ Model Evaluation & Visualization

- Use confusion_matrix, accuracy_score, and f1_score to measure model performance.

- Visualize confusion matrix using seaborn heatmap to assess classification accuracy visually.

## 8️⃣ Run the Application

- Start the Flask server locally:

  - python app.py

- Visit http://localhost:5000 to interact with your web app.
