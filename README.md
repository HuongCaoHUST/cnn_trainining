
# CNN Image Classification on CIFAR-10

This project trains Convolutional Neural Network (CNN) models for image classification using the CIFAR-10 dataset.

## Project Structure

```
.
├── config.yaml              # Configuration file for training parameters
├── data/                    # Directory for dataset
│   └── class_names          # Defines class names for prediction
├── model/                   # Contains CNN model definitions
│   ├── Alexnet.py
│   └── Mobilenet.py
├── src/                     # Source code
│   ├── train.py             # Main training script
│   └── predict.py           # Script for running predictions
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

## Prediction

You can use the `src/predict.py` script to perform predictions on single images or a directory of images. The class names for prediction are loaded from `data/class_names`.

**Arguments:**
*   `--source`: Path to the input image file or a directory containing images.
*   `--weights`: Path to the trained model weights file (`.pth`).
*   `--config`: Path to the configuration file (`config.yaml`). (Default: `config.yaml`)

**Example 1: Predict on a single image**

```bash
python src/predict.py --source test_img.png --weights ./results/run_1/AlexNet_CIFAR10.pth --config config.yaml
```

**Example 2: Predict on a directory of images**

Assuming you have a directory named `my_images` with several `.png` or `.jpg` files:

```bash
python src/predict.py --source data/my_images --weights ./results/run_1/AlexNet_CIFAR10.pth --config config.yaml
```

