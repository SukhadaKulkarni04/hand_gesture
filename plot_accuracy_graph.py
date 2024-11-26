
import matplotlib.pyplot as plt

def plot_accuracy_graph(accuracy, epochs):
    """
    Plots a line graph for accuracy over epochs.
    
    Parameters:
    - accuracy: List of accuracy values per epoch.
    - epochs: Number of epochs.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), accuracy, marker='o', color='b', label='Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss_graph(loss, epochs):
    """
    Plots a line graph for loss over epochs.
    
    Parameters:
    - loss: List of loss values per epoch.
    - epochs: Number of epochs.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), loss, marker='o', color='r', label='Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_comparison_graph(metrics, labels):
    """
    Plots a bar graph comparing different metrics like accuracy, precision, etc.
    
    Parameters:
    - metrics: List of metric values.
    - labels: List of metric names.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(labels, metrics, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.title('Model Performance Metrics Comparison')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()
