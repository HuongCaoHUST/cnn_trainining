import os
import json
import matplotlib.pyplot as plt

def save_results(results, project_root):
    """
    Saves the training history to a JSON file.
    """
    results_path = os.path.join(project_root, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")

def save_plots(history_train_loss, history_val_loss, history_val_accuracy, project_root):
    """
    Generates and saves plots for training/validation loss and validation accuracy.
    """
    # Vẽ biểu đồ Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history_train_loss, label='Training Loss')
    plt.plot(history_val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_path = os.path.join(project_root, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    plt.close() # Đóng figure để giải phóng bộ nhớ
    print(f"Loss plot saved to {loss_plot_path}")

    # Vẽ biểu đồ Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history_val_accuracy, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    acc_plot_path = os.path.join(project_root, 'accuracy_plot.png')
    plt.savefig(acc_plot_path)
    plt.close() # Đóng figure
    print(f"Accuracy plot saved to {acc_plot_path}")
