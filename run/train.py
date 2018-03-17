from datasets.cambridge_landmarks_dataset import CambridgeLandmarksDataset
from models.posenet import PoseNetModified, reprojection_loss
import torch
import torch.utils.data as Data
from torch.autograd import Variable

CONFIG = {
    'learning_rate': 0.0001,
    'epochs': 100,
    'batch_size': 10,
    'use_gpu': False,
}


def train(model=PoseNetModified, config=CONFIG):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_function = reprojection_loss
    train_data_loader = Data.DataLoader(dataset=CambridgeLandmarksDataset, batch_size=config['batch_size'],
                                        shuffle=False)

    for epoch in range(config['epochs']):
        for step, (x, y) in enumerate(train_data_loader):
            b_x = Variable(x)
            b_y = Variable(y)

            output = model(b_x)[0]
            loss = loss_function(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
