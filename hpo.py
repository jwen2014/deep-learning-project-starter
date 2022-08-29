#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms, models

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("BEGIN TESTING")
    model.to("cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print("Test Loss: {test_loss:.4f}")
    logger.info("TESTING ENDED")

def train(model, train_loader, validation_loader, criterion, optimizer, epochs, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("BEGIN TRAINING")
    for i in range(epochs):
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in validation_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()
        
        print(f"Epoch {i}: Train Loss {train_loss:.3f}, Val Loss {val_loss:.3f}")
    logger.info("TRAINING ENDED")


def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    logger.info("INITIALIZE MODEL FOR FINETUNING")
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requies_grad = False
        
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features,  133)
    
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_path = os.path.join(data, "train/")
    validation_path = os.path.join(data, "valid/")
    test_path = os.path.join(data, "test/")
    
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(255),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    testing_transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image_datasets = {
        'train' : torchvision.datasets.ImageFolder(root=train_path, transform=training_transform),
        'valid' : torchvision.datasets.ImageFolder(root=validation_path, transform=testing_transform),
        'test' : torchvision.datasets.ImageFolder(root=test_path, transform=testing_transform)
    }
    
    train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,shuffle=True)
    validation_loader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size, shuffle=False)
    test_loader =  torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False)    

    return train_loader, validation_loader, test_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = net()
    model = model.to(device)
    '''
    TODO: Create your loss and optimizer
    '''
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader, validation_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)
    train(model, train_loader, validation_loader, criterion, optimizer, args.epochs, device)
                     
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader)
    
    '''
    TODO: Save the trained model
    '''
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 2)",
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1.0, 
        metavar="LR", 
        help="learning rate (default: 1.0)"
    )

    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    args = parser.parse_args()
    
    main(args)