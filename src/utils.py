import os
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys # Added for sys.exit

def update_results_csv(epoch, train_loss, val_loss, val_accuracy, save_dir):
    """
    Appends the latest epoch results to a CSV file.
    Creates the file and writes the header on the first call.
    """
    results_path = os.path.join(save_dir, 'results.csv')
    file_exists = os.path.isfile(results_path)

    with open(results_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy'])
        
        writer.writerow([epoch, train_loss, val_loss, val_accuracy])
    # No print statement here to avoid cluttering the epoch log

def save_plots(history_train_loss, history_val_loss, history_val_accuracy, save_dir):
    """
    Generates and saves plots for training/validation loss and validation accuracy
    in the specified directory.
    """
    epochs = range(1, len(history_train_loss) + 1)

    # Vẽ biểu đồ Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history_train_loss, label='Training Loss', marker='o')
    plt.plot(epochs, history_val_loss, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    loss_plot_path = os.path.join(save_dir, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    plt.close() # Đóng figure để giải phóng bộ nhớ
    print(f"Loss plot saved to {loss_plot_path}")

    # Vẽ biểu đồ Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history_val_accuracy, label='Validation Accuracy', marker='o')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    acc_plot_path = os.path.join(save_dir, 'accuracy_plot.png')
    plt.savefig(acc_plot_path)
    plt.close() # Đóng figure
    print(f"Accuracy plot saved to {acc_plot_path}")

def _load_class_names_from_file(file_path):
    """
    Loads class names from a specified file.
    Assumes the file contains a line like: CLASSES = ('class1', 'class2', ...)
    """
    class_names = None
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Use a dictionary to capture the exec'd variables
        exec_globals = {}
        exec(content, exec_globals)
        
        if 'CLASSES' in exec_globals and isinstance(exec_globals['CLASSES'], tuple):
            class_names = exec_globals['CLASSES']
        else:
            raise ValueError(f"Could not find 'CLASSES' tuple in {file_path}")
    except FileNotFoundError:
        print(f"Error: Class names file not found at '{file_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading class names from {file_path}: {e}")
        sys.exit(1)
    return class_names

def count_parameters(model):
    """
    Counts the total number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)