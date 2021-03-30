import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

class CarvanaDatasetTransformsExObject(object):

    def __init__(self, scaled_size):

        self.scaled_size = scaled_size

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(self.scaled_size)
        ])

    def __call__(self, input, target, object):
        input = self.transform(input)
        object = self.transform(object)

        target = torch.unsqueeze(torch.from_numpy(np.array(target, dtype=np.float)), 0)
        target = F.resize(target, self.scaled_size)

        return input, target, object

    def __repr__(self):
        body = [self.__class__.__name__]
        return '\n'.join(body)


class CarvanaDatasetExObject(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, objs_dir, transforms) -> None:
        super().__init__()

        self.images_dir = os.path.expanduser(images_dir)
        self.masks_dir = os.path.expanduser(masks_dir)
        self.objs_dir = os.path.expanduser(objs_dir)

        self.transforms = transforms

        self.images = [file for file in os.listdir(self.images_dir) if os.path.isfile(os.path.join(self.images_dir, file))]
        self.masks = [file for file in os.listdir(self.masks_dir) if os.path.isfile(os.path.join(self.masks_dir, file))]
        self.objs = [file for file in os.listdir(self.objs_dir) if os.path.isfile(os.path.join(self.objs_dir, file))]

    def __getitem__(self, index: int):
        image = Image.open(os.path.join(self.images_dir, self.images[index]))
        mask = Image.open(os.path.join(self.masks_dir, self.masks[index]))
        obj = Image.open(os.path.join(self.objs_dir, self.objs[index]))

        if(self.transforms is not None):
            image, mask, obj = self.transforms(image, mask, obj)

        return image, mask, obj

    def __len__(self) -> int:
        return len(self.images)
