import torch
import numpy as np
from model import LDRN
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import argparse
import os
import time

parser = argparse.ArgumentParser(description='LDRN Single Image Forward Time Measurement',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model_dir', type=str, default='LDRN_KITTI_ResNext101_pretrained_data.pkl')
parser.add_argument('--img_dir', type=str, default='example/9.png')
parser.add_argument('--encoder', type=str, default="ResNext101")
parser.add_argument('--pretrained', type=str, default="KITTI")
parser.add_argument('--norm', type=str, default="BN")
parser.add_argument('--n_Group', type=int, default=32)
parser.add_argument('--reduction', type=int, default=8)
parser.add_argument('--act', type=str, default="ReLU")
parser.add_argument('--lv6', action='store_true')
parser.add_argument('--cuda', default=1)
parser.add_argument('--gpu_num', type=str, default="0，1，2，3")
parser.add_argument('--rank', type=int, default=0, help='node rank for distributed training')
parser.add_argument('--max_depth', type=float, default=80.0, help='max value of depth')

args = parser.parse_args()

# CUDA 设置
if args.cuda and torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    device = torch.device("cuda")
    print("=> Using CUDA")
else:
    device = torch.device("cpu")
    print("=> Using CPU")

# 模型加载
print("=> Loading model...")
Model = LDRN(args).to(device)
Model.load_state_dict(torch.load(args.model_dir, map_location=device))
Model.eval()

# 图像加载与预处理
img = Image.open(args.img_dir).convert("RGB")
img = np.asarray(img, dtype=np.float32) / 255.0
img = img.transpose((2, 0, 1))  # HWC -> CHW
img = torch.from_numpy(img).float()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
img = normalize(img)
img = img.unsqueeze(0).to(device)  # Add batch dim

# 输入尺寸调整
_, _, org_h, org_w = img.shape
if args.pretrained == 'KITTI':
    new_h = 352
else:
    new_h = 432
new_w = int((org_w * (new_h / org_h)) // 16 * 16)
img = F.interpolate(img, (new_h, new_w), mode='bilinear', align_corners=False)

# ✅ 前向推理时间测量（单位：ms）
with torch.no_grad():
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    _, out = Model(img)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()

infer_time_ms = (end_time - start_time) * 1000
print(f"=> Pure forward time for 1 image: {infer_time_ms:.2f} ms")
