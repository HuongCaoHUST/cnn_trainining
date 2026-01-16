import torch
import torchvision
import torchvision.transforms as transforms
import sys
import os

# Thêm đường dẫn thư mục gốc của dự án vào sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from model.Alexnet import AlexNet
# from model.Mobilenet import MobileNet

# =============================================================================
# 1. Thiết lập cấu hình
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Hyperparameters
BATCH_SIZE = 100
MODEL_SAVE_PATH = 'cifar_net.pth' 

# =============================================================================
# 2. Chuẩn bị dữ liệu Test
# =============================================================================
print("Preparing test data...")
# Dữ liệu sẽ được lưu vào thư mục 'data' ở thư mục gốc
data_path = os.path.join(project_root, 'data')

transform = transforms.Compose(
    [transforms.Resize((227, 227)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)
print("Data prepared.")

# =============================================================================
# 3. Tải Model đã lưu
# =============================================================================
print("Loading model...")
# Phải khớp với model đã được dùng để train
model = AlexNet(num_classes=10).to(device)
# model = MobileNet(num_classes=10).to(device)

# Tải model weights từ thư mục gốc của dự án
load_path = os.path.join(project_root, MODEL_SAVE_PATH)
if not os.path.exists(load_path):
    print(f"Error: Model file not found at {load_path}")
    print("Please run train.py first to train and save the model.")
    sys.exit(1)

model.load_state_dict(torch.load(load_path, map_location=device))
print("Model loaded.")

# =============================================================================
# 4. Đánh giá Model trên tập Test
# =============================================================================
print("Evaluating on test set...")
model.eval() 
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')
