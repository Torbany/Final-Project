import pickle
import matplotlib.pyplot as plt

# Load the metrics
with open('models/metrics.pkl', 'rb') as metrics_file:
    metrics = pickle.load(metrics_file)

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
plt.show()
