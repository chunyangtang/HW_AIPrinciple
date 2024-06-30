# CNN for Image Classification

## Usage

First, install the required packages using the following command:

```python
    pip install -r requirements.txt
```

NOTICE that `torchviz` is optional and is only used to plot the model architecture when `models.py` is run as the main file, make sure to use `conda install graphviz` to install `graphviz` for adding the `dot` command to the system path.

You'll also need to place the scene classification data at the `data` folder, place `train_data.csv`, `val_data.csv`, `test_data.csv` and `imgs/` in it.

Then run the following command to reproduce the results:

```python
    python main.py
```

It contains 4 sections:
1. Simple CNN model training, evaluation and testing
2. Improved CNN model (with other settings same as the original) training, evaluation and testing
3. Improved CNN model + Hyperparameter tuning training, evaluation and testing
4. Improved CNN model + Hyperparameter tuning + Data Augmentation training, evaluation and testing
It also save the tesing image as `original.png` and visualized gradcam images as `grad_cam_convx.png` (`x`=1-4) in the root project directory.



# Project Structure
- `main.py` - Main file to run the project
- `models.py` - Definition of `SimpleCNN` & `ImprovedCNN` models, when run as main file, it plots the model architecture in root project directory as `SimpleCNN_model.png` & `ImprovedCNN_model.png`
- `dataloader.py` - Contains `CustomDataset` class and the `load_data` function to load the train, validation and test datasets
- `trainer.py` - Contains `train_model`, `evaluate_model` and `calculate_matrix` functions to train, evaluate and calculate accuracy, precision, recall and f1 score