# CNN Image Classification on CIFAR-10

This project trains Convolutional Neural Network (CNN) models for image classification using the CIFAR-10 dataset.

## Project Structure

```
.
├── config.yaml              # Configuration file for training parameters
├── data/                    # Directory for dataset
├── model/                   # Contains CNN model definitions
│   ├── Alexnet.py
│   └── Mobilenet.py
├── src/                     # Source code
│   ├── train.py             # Main training script
│   └── validation.py        # Script for model validation
├── requirements.txt         # Python dependencies
└── run_train.sh             # Shell script to execute training
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd cnn_trainining
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the dataset:**
    The training script will automatically download the CIFAR-10 dataset if it's not found in the `data` directory.

## Usage

To start the training process, run the following script:

```bash
bash run_train.sh
```

This will execute the `src/train.py` script with the parameters defined in `config.yaml`. You can modify the `config.yaml` file to change the model, learning rate, batch size, number of epochs, etc.
