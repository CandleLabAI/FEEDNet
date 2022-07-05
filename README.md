FEEDNet: A Feature Enhanced Encoder-Decoder LSTM Network for Nuclei Instance Segmentation for Histopathological Diagnosis
============================================

This repository contains the source code of our paper, FEEDNet.

This paper first introduces a novel model, named FEEDNet, for accurately segmenting the nuclei in HE stained WSIs. FEEDNet is an encoder-decoder network that uses LSTM units and “feature enhancement blocks” (FE-blocks). Our proposed FE-block avoids the loss of location information incurred by pooling layers by concatenating the downsampled version of the original image to preserve pixel intensities. FEEDNet uses an LSTM unit to capture multi-channel representations compactly.

![Model Aritechture](reports/figures/feednet.png "Model")

Project Organization
--------------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    |   |   └── dataloader.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    |   |   └── test_model.py
--------

## Get Started
<hr>
Dependencies:

```
pip install -r requirements.txt
```

### (Optional) Conda Environment Configuration

First, create a conda environment
```bash
conda create -n feednet # python=3
source activate feednet
```

Now, add dependencies

Now, you can install the required packages.
```bash
pip install -r requirements.txt
```

### Dataset

We have used CoNSeP, Kumar, and CPM17 dataset which can be downloaded from <a href="https://warwick.ac.uk/fac/sci/dcs/research/tia/data/hovernet/">here for consep</a> and <a href="https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK">here for kumar and cpm17</a>. Download the dataset, unzip it and place in ```data/raw/```. 

To prepare the dataset ready for training, Run following command from ```/src``` directory.

```python data/make_dataset.py```

Above command should prepare Images, and Masks ready for training in ```data/processed``` directory.

To prepare the patches ready for training, Run following command from ```/src``` directory.

```python data/create_patches.py```

### Training

change the hyperparameters and configuration parameters according to need in ```src/config.py```.

To train wscn, Run following command from ```/src``` directory.

```python models/train_model.py``` 

All the trained checkpoints for pre-training as well as full model training will be saved in ```/weights.```

Above command will train binary/multiclass segmentation for given number of epochs.

### Prediction

To train feednet, Run following command from ```/src``` directory.

```python models/predict_model.py --image <path_of_an_image_in_png_format>``` 

Above command will predict the given image and save binary output mask in ```inference/``` directory.

### Test performance

To test feednet with trained model, Run following command from ```/src``` directory.

```python models/test_model.py ``` 

Above command will generate DICE, AJI, DQ, SQ, PQ, Precision, Recall, and F1-Score given in table 3 and 4. 

## Citation
Yet to be updated

## License
<hr>
CC BY-NC-ND 4.0
