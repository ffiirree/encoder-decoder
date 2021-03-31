import argparse
import os
import torch
import torch.nn as nn
from models import UNet, AutoEncoder
from datasets import CarvanaDatasetTransforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='predicted.jpg')
    parser.add_argument('--output-dir', default='logs')

    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(3, 1)
    if device == torch.device('cuda'):
        net = nn.DataParallel(net)
    net.to(device=device)
    
    net.load_state_dict(torch.load(os.path.expanduser(opt.model)))
    net.eval()

    # image
    image = Image.open(os.path.expanduser(opt.input))
    image = CarvanaDatasetTransforms([256, 512]).transform(image)
    image.to(device)

    mask = net(image.unsqueeze(0))[0]
    mask = torch.sigmoid(mask)
    mask = mask.squeeze(0).cpu().detach().numpy()
    mask = mask > 0.5

    mask = Image.fromarray((mask * 255).astype(np.uint8))

    mask.save(os.path.join(opt.output_dir, opt.output))
