import torch
import torch.nn as nn
# Yêu cầu cài thư viện: pip install ultralytics
from ultralytics.nn.modules import Conv, C3k2, SPPF, C2PSA, Detect
from types import SimpleNamespace

class YOLO11(nn.Module):
    def __init__(self, nc=80):
        super().__init__()
        self.nc = nc
        self.conv0 = Conv(3, 16, k=3, s=2)
        self.conv1 = Conv(16, 32, k=3, s=2)
        self.c3k2_2 = C3k2(32, 64, n=1, c3k=False, e=0.25)
        self.conv3 = Conv(64, 64, k=3, s=2)
        self.c3k2_4 = C3k2(64, 128, n=1, c3k=False, e=0.25)
        self.conv5 = Conv(128, 128, k=3, s=2)
        self.c3k2_6 = C3k2(128, 128, n=1, c3k=True)
        self.conv7 = Conv(128, 256, k=3, s=2)
        self.c3k2_8 = C3k2(256, 256, n=1, c3k=True)
        self.sppf9 = SPPF(256, 256, k=5)
        self.c2psa_10 = C2PSA(256, 256, n=1)
        self.up11 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c3k2_13 = C3k2(384, 128, n=1, c3k=False)
        self.up14 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c3k2_16 = C3k2(256, 64, n=1, c3k=False)
        self.conv17 = Conv(64, 64, k=3, s=2)
        self.c3k2_19 = C3k2(192, 128, n=1, c3k=False)
        self.conv20 = Conv(128, 128, k=3, s=2)
        self.c3k2_22 = C3k2(384, 256, n=1, c3k=True)
        self.detect = Detect(nc=nc, ch=[64, 128, 256])
        self.model = [self.detect]
        self.args = SimpleNamespace(
            box=7.5,      # Trọng số loss cho Box (mặc định 7.5)
            cls=0.5,      # Trọng số loss cho Class (mặc định 0.5)
            dfl=1.5,      # Trọng số loss cho DFL (mặc định 1.5)
        )

    def forward(self, x):
        x = self.conv0(x)       # 0
        x = self.conv1(x)       # 1
        x = self.c3k2_2(x)      # 2
        x = self.conv3(x)       # 3
        x = self.c3k2_4(x)      # 4 -> Save P3 (p3_in)
        p3_in = x
    
        x = self.conv5(x)       # 5
        x = self.c3k2_6(x)      # 6 -> Save P4 (p4_in)
        p4_in = x
        
        x = self.conv7(x)       # 7
        x = self.c3k2_8(x)      # 8
        x = self.sppf9(x)       # 9
        x = self.c2psa_10(x)    # 10 -> Save P5 (p5_in)
        p5_in = x
        
        # Layer 11-13
        x = self.up11(p5_in)                # Upsample P5
        x = torch.cat([x, p4_in], dim=1)    # Concat với P4 (Layer 12)
        x = self.c3k2_13(x)                 # Layer 13
        feat_13 = x                         # Lưu lại để dùng sau
        
        # Layer 14-16
        x = self.up14(feat_13)              # Upsample Layer 13
        x = torch.cat([x, p3_in], dim=1)    # Concat với P3 (Layer 15)
        x = self.c3k2_16(x)                 # Layer 16 -> Output P3 cho Head
        p3_out = x
        
        # --- Neck Flow (Bottom-Up) ---
        # Layer 17-19
        x = self.conv17(p3_out)             # Downsample P3
        x = torch.cat([x, feat_13], dim=1)  # Concat với Layer 13 (Layer 18)
        x = self.c3k2_19(x)                 # Layer 19 -> Output P4 cho Head
        p4_out = x
        
        # Layer 20-22
        x = self.conv20(p4_out)             # Downsample P4
        x = torch.cat([x, p5_in], dim=1)    # Concat với P5 gốc (Layer 21)
        x = self.c3k2_22(x)                 # Layer 22 -> Output P5 cho Head
        p5_out = x
        return self.detect([p3_out, p4_out, p5_out])
    
def load_yolo_weights(custom_model, weights_path='yolo11n.pt'):
    print(f"Loading weights from {weights_path}...")
    from ultralytics import YOLO
    original_model = YOLO(weights_path).model
    original_sd = original_model.state_dict()
    custom_sd = custom_model.state_dict()
    
    # Bảng Mapping: Index trong model gốc -> Tên biến trong model custom
    layer_map = {
        0: 'conv0', 1: 'conv1', 2: 'c3k2_2', 3: 'conv3', 4: 'c3k2_4',
        5: 'conv5', 6: 'c3k2_6', 7: 'conv7', 8: 'c3k2_8', 9: 'sppf9', 10: 'c2psa_10',
        13: 'c3k2_13', 16: 'c3k2_16', 17: 'conv17', 19: 'c3k2_19',
        20: 'conv20', 22: 'c3k2_22', 23: 'detect'
    }
    
    tensors_loaded = 0
    update_dict = {}
    
    for k, v in original_sd.items():
        if not k.startswith('model.'): continue
            
        parts = k.split('.')
        idx = int(parts[1]) # Lấy số layer (ví dụ: 0, 10, 23...)
        
        if idx in layer_map:
            name = layer_map[idx]
            suffix = '.'.join(parts[2:]) # phần đuôi (ví dụ: conv.weight)
            
            # Tạo key mới tương ứng với model custom
            new_key = f"{name}.{suffix}"
            
            # Kiểm tra xem key mới có khớp kích thước với custom model không
            if new_key in custom_sd:
                if custom_sd[new_key].shape == v.shape:
                    update_dict[new_key] = v
                    tensors_loaded += 1
                else:
                    # Trường hợp số class khác nhau ở layer Detect
                    print(f"⚠️ Skip {new_key}: Shape mismatch {custom_sd[new_key].shape} vs {v.shape}")

    # Load vào model
    custom_model.load_state_dict(update_dict, strict=False)
    print(f"✅ Successfully loaded {tensors_loaded} tensors!")