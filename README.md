# Supervised Learning Approaches for Predicting Ebola-Human Protein-Protein Interactions

## Environments

### Install via Python (Keras)

# Install PyG
pip install Keras
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib

## Dataset
5 datasets are used.
1. Train dataset - EbolaTrainWhole1.csv
2. Blind Train dataset (1(positive):1(negative)) - EbolaTrainblinddata.csv
3. Blind Test dataset (1(positive):1(negative)) - Ebolatestdataset.csv
4. Oversampled dataset (2(positive):1(negative)) - oversampled.csv
5. Undersampled dataset (1(positive):2(negative)) - undersampled.csv
   
### Offical Dataset
We have shared a snap (reference) of all the datasets as we are not authorized to share the full datasets.

### Preprocessed dataset
We have provided the preprocessed datasets (normalized, standardized) used in the paper.

## Training

All hyper-parameters and training details are provided in the paper.

You can train the model with the following commands:

# Default settings
python ebola_keras_DMLP.py 
python DMLP_Test_Ablations.py
