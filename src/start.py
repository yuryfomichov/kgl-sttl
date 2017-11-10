import glob
import os
import re
import numpy as np
import pandas as pd
import torch as torch
import torch.nn as nn
import torch.optim as optim
from natsort import natsorted
from torch.autograd import Variable
from dataloader import ShipsLoader
from model import ShipModel
from trainer.trainer import ShipsTrainer
from trainer.plugins.saverplugin import SaverPlugin

data_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
def main():
    print('IsCuda', torch.cuda.is_available())
    loss_fn = nn.CrossEntropyLoss().type(data_type)
    loader = ShipsLoader({
        'batch_size': 206
    })
    model = ShipModel().type(data_type)
    optimizer = optim.Adam(model.parameters())
    trainer = ShipsTrainer(model, loader, loss_fn, optimizer, data_type)
    trainer.run(lrs=[1e-3, 1e-4, 1e-5, 1e-6], epochs=[12,12,12,12])
    #checkpoint_data = load_last_checkpoint('checkpoints')
    # if checkpoint_data is not None:
    #     (state_dict, epoch, iteration) = checkpoint_data
    #     trainer.epochs = epoch
    #     trainer.iterations = iteration
    #     trainer.model.load_state_dict(state_dict)


def load_last_checkpoint(checkpoints_path):
    checkpoints_pattern = os.path.join(
        checkpoints_path, SaverPlugin.last_pattern.format('*', '*')
    )
    checkpoint_paths = natsorted(glob.glob(checkpoints_pattern))
    if len(checkpoint_paths) > 0:
        checkpoint_path = checkpoint_paths[-1]
        checkpoint_name = os.path.basename(checkpoint_path)
        match = re.match(SaverPlugin.last_pattern.format(r'(\d+)', r'(\d+)'), checkpoint_name)
        epoch = int(match.group(1))
        iteration = int(match.group(2))
        return (torch.load(checkpoint_path), epoch, iteration)
    else:
        return None

def get_submission():
    (state_dict, epoch, iteration) = load_last_checkpoint('checkpoints')
    model = ShipModel()
    model = model.type(data_type)
    model.load_state_dict(state_dict)
    loader = ShipsLoader({'batch_size': 200, 'shuffle': False})
    df = pd.DataFrame(columns=["is_iceberg"])
    model.eval()
    for x, y in loader.get_submission_loader():
        x_var = Variable(x.type(data_type), volatile=True)
        scores = model(x_var)
        probabitilty = nn.Softmax()(scores).data.cpu().numpy()
        df = df.append(pd.DataFrame(probabitilty[:, 1], columns=["is_iceberg"]))
    df['id'] = loader.submission_data[0]['id'].values
    df = df.reindex_axis(['id'] + ["is_iceberg"], axis=1)
    df.to_csv('submission.csv', index = False)

main()
get_submission()
