import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import sys
import os
import yaml
import numpy as np
from tqdm import tqdm
from torch.utils.data import SubsetRandomSampler

# Thêm đường dẫn thư mục gốc của dự án vào sys.path
# để có thể import các module từ thư mục 'model'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from model.Alexnet import AlexNet
from model.Mobilenet import MobileNet

if __name__ == '__main__':
    # =============================================================================
    # 1. Thiết lập cấu hình và Hyperparameters
    # =============================================================================
    # Tải cấu hình từ file config.yaml
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Hyperparameters từ file config
    NUM_EPOCHS = config['training']['num_epochs']
    BATCH_SIZE = config['training']['batch_size']
    LEARNING_RATE = config['training']['learning_rate']
    MODEL_NAME = config['model']['name']
    MODEL_SAVE_PATH = config['model']['save_path']
    VALIDATION_SPLIT = config['dataset']['validation_split']

    # =============================================================================
    # 2. Chuẩn bị dữ liệu
    # =============================================================================
    print("Preparing data...")
    data_path = os.path.join(project_root, 'data')

    transform = transforms.Compose(
        [transforms.Resize((227, 227)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                                 download=True, transform=transform)

    # Tạo samplers để chia train/validation
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(VALIDATION_SPLIT * num_train))
    
    np.random.seed(42) # Để đảm bảo chia giống nhau mỗi lần chạy
    np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                               sampler=train_sampler, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                                    sampler=valid_sampler, num_workers=2)
    print("Data prepared.")
    print(f"{len(train_indices)} training images, {len(val_indices)} validation images.")


    # =============================================================================
    # 3. Khởi tạo Model, Loss và Optimizer
    # =============================================================================
    print(f"Initializing model: {MODEL_NAME}...")
    if MODEL_NAME == 'AlexNet':
        model = AlexNet(num_classes=10).to(device)
    elif MODEL_NAME == 'MobileNet':
        model = MobileNet(num_classes=10).to(device)
    else:
        print(f"Error: Model '{MODEL_NAME}' not recognized. Exiting.")
        sys.exit()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("Model initialized.")

    # =============================================================================
    # 4. Vòng lặp Training và Validation
    # =============================================================================
    print("Starting Training...")
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        running_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for images, labels in train_progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            val_progress_bar = tqdm(validation_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
            for images, labels in val_progress_bar:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(validation_loader)
        val_accuracy = 100 * correct / total
        
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    print("Finished Training.")

    # =============================================================================
    # 5. Lưu Model
    # =============================================================================
    save_path = os.path.join(project_root, MODEL_SAVE_PATH)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
