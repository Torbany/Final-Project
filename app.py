from flask import Flask, request, jsonify, send_file, render_template
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)

# Load the model, scaler, and metrics
with open('models/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('models/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open('models/metrics.pkl', 'rb') as metrics_file:
    metrics = pickle.load(metrics_file)

# Load the dataset and extract some fraudulent and non-fraudulent transactions
data = pd.read_csv('data/creditcard.csv')
fraudulent_transactions = data[data['Class'] == 1].head(5)
non_fraudulent_transactions = data[data['Class'] == 0].head(5)

fraudulent_transactions_list = fraudulent_transactions.drop('Class', axis=1).values.tolist()
non_fraudulent_transactions_list = non_fraudulent_transactions.drop('Class', axis=1).values.tolist()

@app.route('/')
def home():
    return render_template(
        'index.html', 
        prediction=None, 
        fraudulent_transactions=fraudulent_transactions_list,
        non_fraudulent_transactions=non_fraudulent_transactions_list
    )

@app.route('/predict', methods=['POST'])
def predict():
    features = request.form['features']
    features = np.array([list(map(float, features.split(',')))])
    features = scaler.transform(features)  # Ensure features are scaled
    prediction = model.predict(features)
    prediction_text = 'Fraud' if prediction[0] == 1 else 'Not Fraud'
    return render_template(
        'index.html', 
        prediction=prediction_text, 
        fraudulent_transactions=fraudulent_transactions_list,
        non_fraudulent_transactions=non_fraudulent_transactions_list
    )

@app.route('/metrics', methods=['GET'])
def metrics_endpoint():
    # Create bar plot
    metrics_names = ['Accuracy', 'Precision', 'F1 Score']
    metrics_values = [metrics['accuracy'], metrics['precision'], metrics['f1']]

    plt.figure(figsize=(8, 6))
    plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'red'])
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Model Evaluation Metrics')
    plt.ylim(0, 1)

    # Add text labels on bars
    for i, value in enumerate(metrics_values):
        plt.text(i, value + 0.01, f'{value:.2f}', ha='center', va='bottom')

    plt.savefig('metrics.png')
    return send_file('metrics.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)