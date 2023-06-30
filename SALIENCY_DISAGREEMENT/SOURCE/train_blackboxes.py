import copy
import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch
from torchvision import models
from torch.utils.data import Dataset
import albumentations
import albumentations.pytorch
import multiprocessing

class PneumothoraxImageDataset(Dataset):
    def __init__(self, images, targets, augmentation):
        self.augmentation = augmentation
        self.images = images
        self.targets = targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = np.asarray(Image.open(img_path).convert("RGB"))
        label = self.targets[idx]
        if self.augmentation:
            image = self.augmentation(image=image)['image']
        return image, label

def train_model(model, dataloaders, optimizer, criterion, num_epochs=32, is_inception=False):
    start = time.time()
    val_acc_history = []
    
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-'*10)
        
        # Each epoch's training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()   # Do this if model is in training phase
            else:
                model.eval()    # Do this if model is in validation phase
                
            running_loss = 0
            running_corrects = 0
            
            # Iteration over the data
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Parameter gradients are initialized to 0
                optimizer.zero_grad()
                
                # Forward Pass
                # Getting model outputs and calculating loss
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':      # Special case of inception because InceptionV3 has auxillary outputs as well. 
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                    _, preds = torch.max(outputs, 1)
                    
                    # Backward pass and Optimization in training phase 
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Stats
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # Best model weights are loaded here
    model.load_state_dict(best_model_weights)
    return model, val_acc_history



if __name__ == '__main__':  
    multiprocessing.freeze_support()

    # Params from cli
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-e", "--epochs",type=int, default=30, help="Number epochs to train blackbox models")
    argParser.add_argument("-p", "--path",type=str, help="The dataset folder path")
    argParser.add_argument("-o", "--out",type=str,default='pretrained_weights', help="Output folder path")
    argParser.add_argument("-b", "--batches",type=int, default=32,help="The training batch size")

    args = argParser.parse_args()

    main_path = args.path
    batch_size = args.batches
    epochs = args.epochs
    output_folder = args.out
    num_workers = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 42

    print(device)
    print(f"Training Dataset is retrieve at: {main_path}")
    print(f"Train in {epochs} epochs. {batch_size} images / batch")
    print(f"Output will be stored at: {output_folder}")
    print(f"Number of workers: ", num_workers)
    print(f'Setting everything to seed {seed}')

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    train_data = pd.read_csv(os.path.join(main_path, 'stage_1_train_images.csv'))
    test_data = pd.read_csv(os.path.join(main_path, 'stage_1_test_images.csv'))
    train_data['images'] = train_data['new_filename'].apply(lambda x: os.path.join(main_path, 'png_images', x))
    test_data['images'] = test_data['new_filename'].apply(lambda x: os.path.join(main_path, 'png_images', x))
    images = train_data['images'].tolist()
    targets = train_data['has_pneumo'].tolist()
    print("Train size: ", len(images))
    plt.imshow(Image.open(images[0]))
    
    isExist = os.path.exists(output_folder)
    if not isExist:
        os.makedirs(output_folder) 

    blackbox_configs = { 
        'resnet101': {
            'input_size': (244, 244), 
            'model_constructor': models.resnet101,
            'is_inception': False,
        }, 
        'inceptionv3': {
            'input_size': (299, 299), 
            'model_constructor': models.inception_v3,
            'is_inception': True,
        }
    }

    for model_name, blackbox_config in blackbox_configs.items():
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        resize_to = blackbox_config['input_size']

        aug = albumentations.Compose([
            albumentations.Resize(*resize_to), 
            albumentations.Normalize(mean, std, max_pixel_value=255, always_apply=True), 
            albumentations.pytorch.transforms.ToTensor()
        ])

        train_images, val_images, train_targets, val_targets = train_test_split(images, targets, stratify=targets, train_size=0.9)

        train_dataset = PneumothoraxImageDataset(train_images, train_targets, augmentation=aug)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        val_dataset = PneumothoraxImageDataset(train_images, train_targets, augmentation=aug)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        model = blackbox_config['model_constructor'](weights='IMAGENET1K_V1')
        is_inception = blackbox_config['is_inception']
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        dataloaders_dict = { "train": train_loader, "val": val_loader }
        model, history = train_model(model, dataloaders_dict, optimizer, criterion, epochs, is_inception)
        torch.save(model.state_dict(), f'{output_folder}{model_name}.pth')

