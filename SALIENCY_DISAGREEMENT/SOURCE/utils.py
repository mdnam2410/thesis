# %% [code]
# %% [code]
import os
from torchvision import models, transforms

# ---- Constants -----
DATASET_NAME = 'siim-acr-pneumothorax'

# Dataset
KAGGLE_DATASET_PATH = '/kaggle/input/pneumothorax-chest-xray-images-and-masks'
COLAB_DATASET_PATH = '/content/datasets'
LOCAL_DATASET_PATH = 'pneumothorax-chest-xray-dataset/'

# Checkpoint
KAGGLE_CHECKPOINT_PATH = '/kaggle/input/xai-pretrained-blackbox'
COLAB_CHECKPOINT_PATH = '/content/datasets/'
LOCAL_CHECKPOINT_PATH = 'pretrained_weights/'

def get_path(env):
    dataset_path, checkpoint_path = (LOCAL_DATASET_PATH, LOCAL_CHECKPOINT_PATH) if env == 'local' else \
                                    (KAGGLE_DATASET_PATH, KAGGLE_CHECKPOINT_PATH) if env == 'kaggle' else \
                                    (COLAB_DATASET_PATH, COLAB_CHECKPOINT_PATH)

    dataset_path = os.path.join(dataset_path, DATASET_NAME)
    return dataset_path, checkpoint_path


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