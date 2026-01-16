import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import sys
import os

# Thêm đường dẫn thư mục gốc của dự án vào sys.path
# để có thể import các module từ thư mục 'model'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from model.Alexnet import AlexNet
# from model.Mobilenet import MobileNet

# =============================================================================
# 1. Thiết lập cấu hình và Hyperparameters
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Hyperparameters
NUM_EPOCHS = 15
BATCH_SIZE = 100
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = 'cifar_net.pth'

# =============================================================================
# 2. Chuẩn bị dữ liệu
# =============================================================================
print("Preparing data...")
# Dữ liệu sẽ được lưu vào thư mục 'data' ở thư mục gốc
data_path = os.path.join(project_root, 'data')

transform = transforms.Compose(
    [transforms.Resize((227, 227)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Chỉ tải full training dataset
train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                             download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)
print("Data prepared.")

# =============================================================================
# 3. Khởi tạo Model, Loss và Optimizer
# =============================================================================
print("Initializing model...")
# Chọn model bạn muốn train
model = AlexNet(num_classes=10).to(device)
# model = MobileNet(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print("Model initialized.")

# =============================================================================
# 4. Vòng lặp Training
# =============================================================================
print("Starting Training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}')

print("Finished Training.")

# =============================================================================
# 5. Lưu Model
# =============================================================================
# Lưu model vào thư mục gốc của dự án
save_path = os.path.join(project_root, MODEL_SAVE_PATH)
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
