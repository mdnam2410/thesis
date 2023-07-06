conda create -y --name xai_disagreement python=3.10 pip
conda activate xai_disagreement

conda install -y kaggle cudatoolkit=11.1 -c pytorch -c nvidia
echo "{\"username\":\"yall49\",\"key\":\"7422e489892e03eb63637fd75fa7ea6e\"}" > kaggle.json
mkdir  -p ~/.kaggle
mv kaggle.json ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d vbookshelf/pneumothorax-chest-xray-images-and-masks
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

mkdir -p SALIENCY_DISAGREEMENT/SOURCE/datasets/pneumothorax-chest-xray-images-and-masks
mkdir -p SALIENCY_DISAGREEMENT/SOURCE/datasets/chest-xray-pneumonia
unzip pneumothorax-chest-xray-images-and-masks -d SALIENCY_DISAGREEMENT/SOURCE/datasets/pneumothorax-chest-xray-images-and-masks
unzip chest-xray-pneumonia -d SALIENCY_DISAGREEMENT/SOURCE/datasets/chest-xray-pneumonia

python -m pip install -r SALIENCY_DISAGREEMENT/SOURCE/requirements.txt