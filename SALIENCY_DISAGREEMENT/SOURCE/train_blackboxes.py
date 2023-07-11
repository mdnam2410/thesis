import copy
import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch
from torchvision import datasets, models
from torch.utils.data import Dataset
import albumentations
import albumentations.pytorch
import multiprocessing


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

class ChestXrayImageDataset(Dataset):
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

def createFolder(folder_path):
    isExist = os.path.exists(folder_path)
    if not isExist:
      os.makedirs(folder_path)

def get_dataset(dataset_name, aug, main_path): 
    if dataset_name == 'pneumothorax-chest-xray':
        dataset_path = f"{main_path}/pneumothorax-chest-xray-images-and-masks/siim-acr-pneumothorax"
        train_data = pd.read_csv(os.path.join(dataset_path, 'stage_1_train_images.csv'))
        test_data = pd.read_csv(os.path.join(dataset_path, 'stage_1_test_images.csv'))

        train_data['images'] = train_data['new_filename'].apply(lambda x: os.path.join(dataset_path, 'png_images', x))
        test_data['images'] = test_data['new_filename'].apply(lambda x: os.path.join(dataset_path, 'png_images', x))

        train_images = train_data['images'].tolist()
        train_targets = train_data['has_pneumo'].tolist()
        train_images, val_images, train_targets, val_targets = train_test_split(train_images, train_targets, stratify=train_targets, train_size=0.9)
        train_dataset = ChestXrayImageDataset(train_images, train_targets, augmentation=aug)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_dataset = ChestXrayImageDataset(val_images, val_targets, augmentation=aug)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        test_dataset = ChestXrayImageDataset(test_data['images'].tolist(), test_data['has_pneumo'].tolist(), augmentation=aug)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        return train_loader, val_loader, test_loader
    elif dataset_name == 'pneumonia-chest-xray':        
        dataset_path = f"{main_path}/chest-xray-pneumonia/chest_xray"
        aug_adaptor = lambda image: aug(image=np.asarray(image))['image']
        train_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform = aug_adaptor)
        test_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'test'), transform = aug_adaptor)
        val_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'val'), transform = aug_adaptor)

        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size,shuffle = True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_dataset,batch_size = batch_size,shuffle = False, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = batch_size,shuffle = False, num_workers=0)
        return train_loader, val_loader, test_loader
    return ()

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


def train(model_name, dataset_name, main_path):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    blackbox_config = blackbox_configs[model_name]
    resize_to = blackbox_config['input_size']

    aug = albumentations.Compose([
        albumentations.Resize(*resize_to), 
        albumentations.Normalize(mean, std, max_pixel_value=255, always_apply=True), 
        albumentations.pytorch.transforms.ToTensor()
    ])
    train_loader, val_loader, _ = get_dataset(dataset_name, aug, main_path)
    model = blackbox_config['model_constructor'](pretrained=True)
    is_inception = blackbox_config['is_inception']
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    dataloaders_dict = { "train": train_loader, "val": val_loader }
    model, _ = train_model(model, dataloaders_dict, optimizer, criterion, epochs, is_inception)

    output_dir = f'{output_folder}/{dataset_name}'
    createFolder(output_dir)
    torch.save(model.state_dict(), f'{output_dir}/{model_name}.pth')

if __name__ == '__main__':  
    multiprocessing.freeze_support()

    # Params from cli
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-e", "--epochs",type=int, default=30, help="Number epochs to train blackbox models")
    argParser.add_argument("-p", "--path",type=str, help="The dataset folder path")
    argParser.add_argument("-o", "--out",type=str,default='SALIENCY_DISAGREEMENT/SOURCE/pretrained_weights', help="Output folder path")
    argParser.add_argument("-b", "--batches",type=int, default=32,help="The training batch size")
    argParser.add_argument("-d", "--dataset",type=str, default="pneumonia", help="The name of the dataset used in training phase")
    argParser.add_argument("-m", "--model",type=str, default="res101", help="The name of the dataset used in training phase")

    args = argParser.parse_args()

    main_path = args.path
    batch_size = args.batches
    epochs = args.epochs
    output_folder = args.out
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 42
    model_name = args.model
    dataset_name = args.dataset

    print(device)
    print(f"Training Dataset is retrieve at: {main_path}")
    print(f"Train in {epochs} epochs. {batch_size} images / batch")
    print(f"Output will be stored at: {output_folder}")
    print(f'Setting everything to seed {seed}')

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    createFolder(output_folder)
    train(model_name, dataset_name, main_path)


