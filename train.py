"""
Trains a PyTorch image classification model.
"""

# Importations
import argparse
import os
import torch
from torch import nn
from torchvision import transforms
from timeit import default_timer as timer
import random
from scripts import data_setup, engine, model_builder, utils, get_data

# Setup Parser
parser = argparse.ArgumentParser(prog='Model training', description='Train a model on custom data.')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--color_channels', type=int, default=3)
parser.add_argument('--hidden_units', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--img_size', type=int, default=64)

parser.add_argument('--local', action='store_true')
parser.add_argument('--url', type=str, default="data/dataset")

# Setup HyperParameters
NUM_EPOCHS = parser.parse_args().epochs
BATCH_SIZE = parser.parse_args().batch_size
INPUT_SHAPE = parser.parse_args().color_channels
HIDDEN_UNITS = parser.parse_args().hidden_units
LEARNING_RATE = parser.parse_args().lr
IMG_SIZE = (parser.parse_args().img_size,parser.parse_args().img_size)

IS_LOCAL = parser.parse_args().local
URL = parser.parse_args().url

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
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Start the timer
start_time = timer()

# Train model using engine.py
model_results = engine.train(model=model,
                             train_dataloader=train_dataloader,
                             test_dataloader=test_dataloader,
                             loss_fn=loss_fn,
                             optimizer=optimizer,
                             epochs=NUM_EPOCHS,
                             device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# Save the model using utils.py
models_dir = "models"
results_dir = "results"
model_name = f"{model.__class__.__name__}_model_{NUM_EPOCHS}_ep_{LEARNING_RATE}_lr_{BATCH_SIZE}_bs_{HIDDEN_UNITS}_hu_{INPUT_SHAPE}_cc_{parser.parse_args().img_size}_img#{random.randint(1, 100)}"

utils.save_model(model=model, target_dir=models_dir, model_name=model_name+".pth")
utils.save_results(results=model_results, target_dir=results_dir, model_name=model_name+".csv")
