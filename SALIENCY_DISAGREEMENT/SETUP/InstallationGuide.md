# System Installation

## Requirements
The system requires the following conditions to be available on the machine:
- Internet connection
- Python 3 with `pip` package manager
- `conda` package manager

The [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) command can be installed with either Anaconda or miniconda.

### For Windows
The installation requires bash scripts being able to be executed. Command Prompt or Powershell are not able to run bash script so we recommend using other solutions for Windows. We recommend:
- Git bash
- Windows Subsystem for Linux

## Installation
Assume we are at the project's top-level folder, run the following command
```
$ ./SALIENCY_DISAGREEMENT/SETUP/setup_env.sh
```

The script `setup_env.sh` sets up the following:
- Create a new conda environment named `xai_disagreement`
- Install all required packages for training, generating explanations and visualization
- Download training datasets

After installation we should see a new directory `SALIENCY_DISAGREEMENT/SOURCE/dataset` that contains the training and test data. There will be two subdirectories corresponding two datasets, namely `pneumothorax-chest-xray-dataset` and `chest-xray-pneumonia`.

## Usage
Make sure to use the system with the `xai_disagreement` environment activated, using:
```
$ conda activate xai_disagreement
```

The system is run in three steps:
1. Training
2. Generating explanations
3. Visualization

### Step 1: Training
To train black boxes, run:
```
$ ./SALIENCY_DISAGREEMENT/SOURCE/train.sh
```

After training the output black box checkpoints will be stored in the directory `pretrained_weights`. These checkpoints are required for step 2. We can skip this step if you are satisfied with the current pretrained models.

### Step 2: Generating explanations
Launch Jupyter notebook using:
```
$ jupyter notebook
```

Run all cells within the notebook `src/generate_explanation.ipynb` to compute the explanations for the black boxes on the test set data. Computed explanations are stored in the directory `explanations` and are required for visualization in step 3.

### Step 3: Visualization
Run all cells within the notebook `src/visualization.ipynb` to compute the disagreement heatmaps. The figures of the heatmaps are stored in `figures`. These figures are the results of our thesis.
