conda create --name xai_disagreement python=3.10
conda activate xai_disagreement

conda install kaggle cudatoolkit=11.1 -c pytorch -c nvidia
read -p "Your Kaggle user name: " username
read -p "Your Kaggle API key: " api
echo "{\"username\":\"$username\",\"key\":\"$api\"}" > kaggle.json
mkdir  -p ~/.kaggle
mv kaggle.json ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d vbookshelf/pneumothorax-chest-xray-images-and-masks
unzip -d pneumothorax-chest-xray-dataset pneumothorax-chest-xray-images-and-masks
pip install -r train_requirements.txt
python train_blackboxes.py --path pneumothorax-chest-xray-dataset/siim-acr-pneumothorax