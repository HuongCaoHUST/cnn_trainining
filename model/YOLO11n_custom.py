import torch
import torch.nn as nn
from ultralytics.nn.modules import (
    C2PSA,
    SPPF,
    C3k2,
    Conv,
)

class YOLO11Custom(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(Conv(c1=3, c2=16, k=3, s=2))
        self.layers.append(Conv(c1=16, c2=32, k=3, s=2))
        self.layers.append(C3k2(c1=32, c2=64, n=1, c3k=False, e=0.25))
        self.layers.append(Conv(c1=64, c2=64, k=3, s=2))
        self.layers.append(C3k2(c1=64, c2=128, n=1, c3k=False, e=0.25))
        self.layers.append(Conv(c1=128, c2=128, k=3, s=2))
        self.layers.append(C3k2(c1=128, c2=128, n=1, c3k=True))
        self.layers.append(Conv(c1=128, c2=256, k=3, s=2))
        self.layers.append(C3k2(c1=256, c2=256, n=1, c3k=True))
        self.layers.append(SPPF(c1=256, c2=256, k=5))
        self.layers.append(C2PSA(c1=256, c2=256, n=1))
        self.output_indices = [4, 6, 10]

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.output_indices:
                outputs.append(x)
        return outputs
    
def load_pretrained_weights(custom_model, pt_path='yolo11n.pt'):
    print(f"Loading weights from {pt_path}...")
    from ultralytics import YOLO
    full_model = YOLO(pt_path).model
    custom_dict = custom_model.state_dict()
    pretrained_dict = full_model.state_dict()
    
    new_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('model.'):
            layer_idx = int(k.split('.')[1])
            if layer_idx <= 10:
                new_key = k.replace('model.', 'layers.', 1)
                if new_key in custom_dict:
                    new_dict[new_key] = v
    custom_model.load_state_dict(new_dict, strict=True)
    print(f"Loaded {len(new_dict)} tensors successfully!")