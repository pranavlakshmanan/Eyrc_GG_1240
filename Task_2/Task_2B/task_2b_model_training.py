import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pathlib
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torchvision.models import ResNet18_Weights
from PIL import Image
from torch.autograd import Variable
import pandas as pd

transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
training_set = ImageFolder("Task2_dataset//training", transform=transform)
val_size = 120
train_size = len(training_set) - val_size
train_ds, val_ds = random_split(training_set, [train_size, val_size])
len(train_ds), len(val_ds)
batch_size = 4
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=4, shuffle=True, num_workers=0)
root = pathlib.Path("Task2_dataset//training")
class_names = sorted([j.name.split('/')[-1] for j in root.iterdir()])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is set to", device)
print(class_names)


def train_model(model, criterion, optimizer, scheduler, num_epochs):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()
                epoch_loss = running_loss / train_size
                epoch_acc = running_corrects.double() / train_size
            else:
                epoch_loss = running_loss / val_size
                epoch_acc = running_corrects.double() / val_size

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model

def prediction(transformer):
    headers = ["Acutal Image Name", "   Output Image Name"]
    data = []
    for img in test_data:
        img_path = "Task2_dataset//testing/"+img+".jpeg"
        image = Image.open(img_path)
        image_tensor = transformer(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        output = model_conv(input)
        index = output.data.numpy().argmax()
        pred = class_names[index]
        data.append([img, pred])
    print(pd.DataFrame(data, columns=headers))

if __name__ == "__main__":
    eps = 3
    ss=7
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=ss, gamma=1)
    model = train_model(model, criterion, optimizer,
                        step_lr_scheduler, num_epochs=eps)
    model_conv = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model_conv.parameters():
        param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 5)
    model_conv = model_conv.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=ss, gamma=1)

    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, num_epochs=eps)
    torch.save(model, "model_conv.pth")

    test_data = ["building1", "building2", "combat1", "combat2",
                 "fire1", "fire2", "military1", "military2", "rehab1", "rehab2"]

    prediction(transform)
