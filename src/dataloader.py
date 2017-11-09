import torch.utils.data.dataloader as dataloader
import torch as torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataset import ShipsDataset
import os
from PIL import Image


class ShipsLoader(object):
    def __init__(self, params):
        self.validation_size = params.get("validation_size",0.0999)
        self.batch_size = params.get("batch_size", 200)
        self.num_workers = params.get("num_workers", 8 if torch.cuda.is_available() else 0)
        self.shuffle = params.get("shuffle", True)
        self.data_dir = 'data'
        self.train_folder = 'train'
        self.test_folder = 'test'
        self._load_data()
        self._load_submission_data()

    def get_train_loader(self):
        return self._get_loader(self.train_data, True, False)

    def get_val_loader(self):
        return self._get_loader(self.val_data, False, False)

    def get_test_loader(self):
        return self._get_loader(self.test_data, False, False)

    def get_submission_loader(self):
         return self._get_loader(self.submission_data, False, True)

    def _get_loader(self, data, drop_last = False, is_test = True):
        loader = dataloader.DataLoader(ShipsDataset(data[0], data[1], is_train= not is_test),
                                       batch_size=self.batch_size,
                                       shuffle=self.shuffle,
                                       num_workers=self.num_workers,
                                       drop_last=drop_last,
                                       pin_memory=torch.cuda.is_available())
        return loader

    def _load_data(self):
        data = pd.read_json('data/train.json')
        #self._combine_image(data);
        ids = data
        labels = data['is_iceberg'].values;
        (X_train, x_temp, y_train, y_temp) = train_test_split(ids, labels, test_size=self.validation_size,stratify=labels)
        self.train_data = (X_train, y_train)

        (X_val, X_test, y_val, y_test) = train_test_split(x_temp, y_temp, test_size=0.01)
        self.val_data = (X_val, y_val)
        self.test_data = (X_test, y_test)

    def _load_submission_data(self):
        data = pd.read_json('data/test.json')
        labels = np.zeros(data.shape[0])
        self.submission_data = (data, labels)

    def _combine_image(self, df):
        # band_1 = list(map(lambda x: np.array(x).reshape((75, 75)), df['band_1']))
        # band_2 = list(map(lambda x: np.array(x).reshape((75, 75)), df['band_2']))
        # images = np.add(band_1, band_2)
        # images -= images.min(axis=0)
        # images /= images.max(axis=0)
        # images *= 255
        # images = images.astype(np.uint8)
        # return images;
        for ix, row in df.iterrows():
            img1 = np.array(row['band_1']).reshape((75, 75, 1))
            img2 = np.array(row['band_2']).reshape((75, 75, 1))
            img3 = img1/img2
            img1 = img1 - img1.min()
            img1 = img1 / img1.max() * 255
            img2 = img2 - img2.min()
            img2 = img2 / img2.max() * 255
            img3 = img3 - img3.min()
            img3 = img3 / img3.max() * 255
            result = np.dstack((img1, img2, img3))
            result = result.astype(np.uint8)
            im = Image.fromarray(result)
            if row['is_iceberg'] == 0:
                im.save("data/train/0/{}.jpeg".format(row['id']))
            elif row['is_iceberg'] == 1:
                im.save("data/train/1/{}.jpeg".format(row['id']))
