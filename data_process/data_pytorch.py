import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy, Loss, Precision, Recall
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader


# %% Define the dataset

class GammaHadron(Dataset):
    def __init__(self, train):
        if train == True:
            with open('/dev/shm/data/hadron_numpy_train.pkl', 'rb') as f:
                hadron = pickle.load(f)

            with open('/dev/shm/data/gamma_numpy_train.pkl', 'rb') as f:
                gamma = pickle.load(f)
        else:
            with open('/dev/shm/data/hadron_numpy_test.pkl', 'rb') as f:
                hadron = pickle.load(f)

            with open('/dev/shm/data/gamma_numpy_test.pkl', 'rb') as f:
                gamma = pickle.load(f)

        num_hadron = hadron.shape[0]
        num_gamma = gamma.shape[0]
        hadron = np.reshape(hadron, (num_hadron, 1, 101, 101))
        gamma = np.reshape(gamma, (num_gamma, 1, 101, 101))
        torc_hadron = torch.from_numpy(hadron).float()
        torc_gamma = torch.from_numpy(gamma).float()
        self.x = torch.cat((torc_hadron, torc_gamma), 0)

        hadron0 = torch.zeros(num_hadron)
        gamma1 = torch.ones(num_gamma)
        self.y = torch.cat((hadron0, gamma1)).long()

        self.len = self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item, :, :], self.y[item]

    def __len__(self):
        return self.len


# %% Load the datasets

train_dataset = GammaHadron(train=True)
test_dataset = GammaHadron(train=False)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=250,
                          shuffle=True,
                          )

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=250,
                         shuffle=True,
                         )


# %% Model Definition
# model = resnet18(pretrained=False, num_classes=2)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(40000, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


model = ConvNet()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


# %% TensorboardX setting

def create_summary_writer(model, data_loader):
    writer = SummaryWriter(comment='--Simple MNIST-like')
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
        print("model graph saved")
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


writer = create_summary_writer(model, train_loader)

# %% Ignite setting
device = 'cuda'
log_interval = 10
trainer = create_supervised_trainer(model=model,
                                    optimizer=optimizer,
                                    loss_fn=loss,
                                    device=device)
evaluator = create_supervised_evaluator(model,
                                        metrics={'accuracy': CategoricalAccuracy(),
                                                 'loss': Loss(F.nll_loss),
                                                 'precision': Precision(),
                                                 'recall': Recall()},
                                        device=device)


@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(engine):
    iter = (engine.state.iteration - 1) % len(train_loader) + 1
    if iter % log_interval == 0:
        print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
              "".format(engine.state.epoch, iter, len(train_loader), engine.state.output))
    writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)
    writer.add_scalar("Loss", engine.state.output, engine.state.iteration)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    evaluator.run(train_loader)
    # print(evaluator.state.metrics)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_nll = metrics['loss']
    print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(engine.state.epoch, avg_accuracy, avg_nll))
    writer.add_scalar("training/avg_loss", avg_nll, engine.state.epoch)
    writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(engine):
    evaluator.run(test_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_nll = metrics['loss']
    precision = metrics['precision']
    recall = metrics['recall']
    print("Test Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(engine.state.epoch, avg_accuracy, avg_nll))
    writer.add_scalar("test/avg_loss", avg_nll, engine.state.epoch)
    writer.add_scalar("test/avg_accuracy", avg_accuracy, engine.state.epoch)
    writer.add_scalar("test/precision", precision[0], engine.state.epoch)
    writer.add_scalar("test/recall", recall[0], engine.state.epoch)


# Let the show begin
trainer.run(train_loader, 200)
