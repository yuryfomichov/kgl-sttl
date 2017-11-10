import torch.utils.data as data
import numpy as np
import transforms.transforms as transforms
from PIL import Image

class ShipsDataset(data.Dataset):
    def __init__(self, data, target, is_train):
        self.data = data
        self.targets = target
        self.is_train = is_train

    def __getitem__(self, index):
        band1 = np.array(self.data.iloc[index]['band_1']).reshape((75, 75, 1))
        band2 = np.array(self.data.iloc[index]['band_2']).reshape((75, 75, 1))
        img1 = band1
        img2 = band2
        img3 = (img1 + img2) / 2
        img1 = img1 - img1.min()
        img1 = img1 / img1.max() * 255
        img2 = img2 - img2.min()
        img2 = img2 / img2.max() * 255
        img3 = img3 - img3.min()
        img3 = img3 / img3.max() * 255
        result = np.dstack((img1, img2, img3))
        result = result.astype(np.uint8)
        img_transformator = self._train_image_transform() if self.is_train else self._val_image_transform()
        result = img_transformator(Image.fromarray(result))
        return result, self.targets[index]

    def __len__(self):
        return len(self.targets)

    def _train_image_transform(self):
        transform = transforms.Compose([
            transforms.Resize(150),
            transforms.RandomCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        return transform

    def _val_image_transform(self):
        transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        return transform
