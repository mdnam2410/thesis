conda create --name xai_disagreement python=3.10
conda activate xai_disagreement

conda install kaggle cudatoolkit=11.1 -c pytorch -c nvidia
echo "{\"username\":\"yall49\",\"key\":\"7422e489892e03eb63637fd75fa7ea6e\"}" > kaggle.json
mkdir  -p ~/.kaggle
mv kaggle.json ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d vbookshelf/pneumothorax-chest-xray-images-and-masks
pip install -r train_requirements.txt
unzip -d pneumothorax-chest-xray-dataset pneumothorax-chest-xray-images-and-masks
pip install -r requirements.txt
