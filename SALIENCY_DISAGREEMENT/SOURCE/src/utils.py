import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class PneumothoraxImageDataset(Dataset):
    """Loader for the Pneumothorax Chest X-ray Images and Masks dataset
    """

    def __init__(self, filenames, images, targets, masks, transform, mask_transform):
        self.filenames = filenames
        self.masks = masks
        self.images = images
        self.targets = targets
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = np.asarray(Image.open(self.images[idx]).convert('RGB'))
        mask = np.asarray(Image.open(self.masks[idx]).convert('RGB'))
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return filename, image, mask, label


# ---- Constants -----
DS_PNEUMOTHORAX = 'Pneumothorax'
DS_PNEUMONIA = 'Pneumonia'

# Black boxes
BLACK_BOX_NAMES = ['InceptionV3', 'ResNet']

# Explanation methods
METHOD_NAMES = ['DeepLift', 'GradientShap', 'GuidedBackprop',
                'GuidedGradCam', 'IntegratedGradients', 'LRP', 'Occlusion', 'Saliency']

# ----- Directories -----
# Datasets
KAGGLE_DATASET_PATH = '/kaggle/input/'
COLAB_DATASET_PATH = '/content/datasets'
LOCAL_DATASET_PATH = '../datasets/'

# Checkpoints
KAGGLE_CHECKPOINT_PATH = '/kaggle/input/xai-pretrained-blackbox'
COLAB_CHECKPOINT_PATH = '/content/datasets/'
LOCAL_CHECKPOINT_PATH = 'pretrained_weights/'

# Serialized explanations
SAVED_EXPLANATIONS_DIR = '../explanations'

# Figues
FIGURES_PATH = '../figures'

# ---- For transforming images ----
INPUT_SIZE = (128, 128)

resize_image = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
])

transform = transforms.Compose([
    transforms.ToTensor(),
    resize_image,
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

mask_transform = transforms.Compose([
    transforms.ToTensor(),
    resize_image,
])


def get_path(env):
    dataset_path, checkpoint_path = (LOCAL_DATASET_PATH, LOCAL_CHECKPOINT_PATH) if env == 'local' else \
                                    (KAGGLE_DATASET_PATH, KAGGLE_CHECKPOINT_PATH) if env == 'kaggle' else \
                                    (COLAB_DATASET_PATH, COLAB_CHECKPOINT_PATH)
    return dataset_path, checkpoint_path


def get_dataset_loader(env, dataset, batch_size):
    dataset_path, _ = get_path(env)
    ds_loader = None

    if dataset == DS_PNEUMOTHORAX:
        dataset_path = os.path.join(
            dataset_path, 'pneumothorax-chest-xray-images-and-masks/siim-acr-pneumothorax')

        test_data = pd.read_csv(os.path.join(
            dataset_path, 'stage_1_test_images.csv'))
        test_data['images'] = test_data['new_filename'].apply(
            lambda x: os.path.join(dataset_path, 'png_images', x))
        test_data['masks'] = test_data['new_filename'].apply(
            lambda x: os.path.join(dataset_path, 'png_masks', x))

        filenames = test_data['new_filename'].tolist()
        images = test_data['images'].tolist()
        masks = test_data['masks'].tolist()
        targets = test_data['has_pneumo'].tolist()
        dataset = PneumothoraxImageDataset(
            filenames, images, targets, masks, transform, mask_transform)
        ds_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    elif dataset == DS_PNEUMONIA:
        dataset_path = os.path.join(
            dataset_path, 'chest-xray-pneumonia/chest_xray')
        test_dataset = datasets.ImageFolder(os.path.join(
            dataset_path, 'test'), transform=transform)
        ds_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return ds_loader

def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
