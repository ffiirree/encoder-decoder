import argparse
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from models import UNet, AutoEncoder
from datasets import CarvanaDataset, CarvanaDatasetTransforms
from metrics import dice_coeff
from utils import tqdm_with_logging_redirect, make_logger
from torch.utils.data import DataLoader, random_split

logger = make_logger('unet', 'logs')

def validate(net, loader, device):
    net.eval()

    loss = 0
    with tqdm(total=len(loader), dynamic_ncols=True, desc="validation", unit="batch", leave=False) as pbar:
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            with torch.no_grad():
                output = net(images)

            output = torch.sigmoid(output)
            output = (output > 0.5).float()

            loss += dice_coeff(output, masks).item()
            pbar.update()

    net.train()
    return loss / len(loader)

def train_net(net, device, train_loader, val_loader, epochs, optimizer, criterion, clip_grad=True):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=6)
    for epoch in range(epochs):
        net.train()
        with tqdm_with_logging_redirect(total=n_train, dynamic_ncols=True, desc=f'epoch {epoch + 1}/{epochs}', unit='img', logger=logger) as pbar:
            for index, (images, masks) in enumerate(train_loader):
                images, masks = images.to(device), masks.to(device)

                output = net(images)
                loss = criterion(output, masks)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()

                if clip_grad:
                    nn.utils.clip_grad_value_(net.parameters(), 0.1)

                optimizer.step()

                pbar.update(images.shape[0])

                if (index + 1) % (n_train // (10 * images.shape[0])) == 0:
                    val_score = validate(net, val_loader, device)
                    scheduler.step(val_score)
                    logger.info(f'[validation {epoch}/{epochs}@{index:>3d}] lr: {optimizer.param_groups[0]["lr"]}, dice coeff: {val_score}' )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--image-size', nargs='+', default=[256, 512])
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--n_val', type=int, default=512)
    parser.add_argument('--output-dir', default='logs')
    parser.add_argument('--clip-grad', type=bool, default=True)

    opt = parser.parse_args()
    logger.info(opt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')

    dataset = CarvanaDataset(
        '~/data/datasets/carvana/train',
        '~/data/datasets/carvana/train_masks',
        transforms=CarvanaDatasetTransforms(opt.image_size)
    )

    # torch.manual_seed(0)
    n_train = len(dataset) - opt.n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, opt.n_val], generator=torch.Generator().manual_seed(0))

    logger.info(f'dataset: {len(dataset)}, train: {len(dataset) - opt.n_val}, val: {opt.n_val}')

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=True, drop_last=True)

    net = UNet(3, n_classes=1)
    if device == torch.device('cuda'):
        net = nn.DataParallel(net, device_ids=[0,1,2,3])
        logger.info(f'use gpu: {net.device_ids}')
    net.to(device=device)

    # optimizer = optim.RMSprop(net.parameters(), lr=opt.lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()

    train_net(
        net=net,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=opt.epochs,
        optimizer=optimizer,
        criterion=criterion,
        clip_grad=opt.clip_grad
    )

