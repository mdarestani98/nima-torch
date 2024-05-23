import argparse
import os
from collections import deque

import cv2
import tqdm
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from model.model import NIMA


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='path to pretrained model')
parser.add_argument('--video', type=str, help='path to video', default=None)
parser.add_argument('--temp', type=str, help='path to temp folder', default='./temp')
parser.add_argument('--skip', type=int, help='skip frames', default=5)
parser.add_argument('--fps', type=float, help='frames per second', default=None)
parser.add_argument('--save_video', type=str, help='output file to store video with predictions')
args = parser.parse_args()

assert args.video and os.path.exists(args.video), 'Please provide a video file'
if not os.path.exists(args.temp):
    os.makedirs(args.temp)

base_model = models.efficientnet_b0()
model = NIMA(base_model)

try:
    model.load_state_dict(torch.load(args.model))
    print('successfully loaded model')
except:
    raise

seed = 42
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

vidcap = cv2.VideoCapture(args.video)
if not args.fps:
    args.fps = vidcap.get(cv2.CAP_PROP_FPS)
success, image = vidcap.read()
count = 0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.save_video, fourcc, args.fps, (image.shape[1], image.shape[0]))
pbar = tqdm.tqdm(total=int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)))
past_second = deque(maxlen=int(args.fps))
font = cv2.FONT_HERSHEY_SIMPLEX
org = (10, 100)
font_scale = 3
thickness = 3
text_size = cv2.getTextSize('8.98', font, font_scale, thickness)[0]
top_left = (org[0], org[1] - text_size[1] - 10)
bottom_right = (org[0] + text_size[0] + 10, org[1] + 10)
while success:
    if count % args.skip == 0:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        imt = test_transform(pil_img).unsqueeze(dim=0).to(device)
        output = model(imt).view(10,).tolist()
        mean = sum(i * e for i, e in enumerate(output, 1))
        std = sum(e * (i - mean) ** 2 for i, e in enumerate(output, 1))
        past_second.append(mean)
    color = (0, 0, 255) if min(past_second) < 7 else (0, 255, 0)
    text = f'{sum(past_second) / len(past_second):.3}'
    cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
    cv2.putText(image, text, org, font, font_scale, (0, 0, 0), thickness)
    out.write(image)
    success, image = vidcap.read()
    count += 1
    pbar.update(1)
vidcap.release()
out.release()
pbar.close()
