import argparse
import os

from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from model.model import NIMA

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='path to pretrained model')
parser.add_argument('--dir', type=str, help='path to folder containing images', default=None)
parser.add_argument('--image', type=str, help='path to image', default=None)
parser.add_argument('--save', type=str, help='output file to store predictions')
args = parser.parse_args()

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

if args.dir:
    test_images = os.listdir(args.dir)
    test_images = [os.path.join(args.dir, img) for img in test_images]
elif args.image:
    test_images = [args.image]
else:
    raise ValueError('Either --dir or --image must be provided')

for img in test_images:
    im = Image.open(img)
    im = im.convert('RGB')
    imt = test_transform(im).unsqueeze(dim=0).to(device)
    output = model(imt).view(10,).tolist()
    mean = sum(i * e for i, e in enumerate(output, 1))
    std = sum(e * (i - mean) ** 2 for i, e in enumerate(output, 1))
    print(f'{img}: {mean}, {std}')
    if args.save:
        with open(args.save, 'a') as f:
            f.write(f'{img}, {mean}, {std}\n')
