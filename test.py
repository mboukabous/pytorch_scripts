"""
Test a PyTorch classification model
"""

#Importations
import argparse
import os
import torch
from torch import nn
from torchvision import transforms
from timeit import default_timer as timer
import random
from tqdm.auto import tqdm
from scripts import data_setup, model_builder, utils, get_data

# Setup Parser
parser = argparse.ArgumentParser(prog='Model testing', description='Test a model on custom data.')

parser.add_argument('--model', type=str, default="models/model.pth")
parser.add_argument('--local', action='store_true')
parser.add_argument('--url', type=str, default="data/dataset")

# Setup HyperParameters
MODEL_PATH = parser.parse_args().model
IS_LOCAL = parser.parse_args().local
URL = parser.parse_args().url

# Get HyperParameters
BATCH_SIZE = int(MODEL_PATH.split("_bs_")[0].split("_")[-1])
HIDDEN_UNITS = int(MODEL_PATH.split("_hu")[0].split("_")[-1])
IMG_SIZE = (int(MODEL_PATH.split("_img")[0].split("_")[-1]), int(MODEL_PATH.split("_img")[0].split("_")[-1]))
INPUT_SHAPE = int(MODEL_PATH.split("_cc_")[0].split("_")[-1])

# Setup directories
if IS_LOCAL:
  train_dir, test_dir = get_data.local_data(URL)
else:
  train_dir, test_dir = get_data.download_data(URL)

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create or import transforms here

# Create dataLoaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               batch_size=BATCH_SIZE,
                                                                               img_size=IMG_SIZE)
                                                                               #train_transform=data_transform
                                                                               #test_transform=data_transform

# Recreate an instance of TinyVGG
model = model_builder.TinyVGG(input_shape=INPUT_SHAPE,
                              hidden_units=HIDDEN_UNITS,
                              output_shape=len(class_names)).to(device)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()

# Load the model parameters
model = utils.load_model(model=model, model_path=MODEL_PATH)

# Test loop
test_time_start = timer()
loss, acc = 0, 0
model.eval()

with torch.inference_mode():
  for X, y in tqdm(test_dataloader):
    # Device agnostic code
    X = X.to(device)
    y = y.to(device)

    # Forward pass and calculate the metrics
    y_pred = model(X)
    loss += loss_fn(y_pred, y)

    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    acc += (y_pred_class == y).sum().item()/len(y_pred)

  # calculate total loss and accuracy (average)
  loss /= len(test_dataloader)
  acc /= len(test_dataloader)

# End the timer and print out how long it took
test_time_end = timer()

# Print results
print(f"[INFO] Model: {model.__class__.__name__}")
print(f"[INFO] Model loss: {loss:.4f}")
print(f"[INFO] Model accuracy: {acc*100:.4f}%")
print(f"[INFO] Model execution time: {test_time_end-test_time_start:.3f} seconds")
