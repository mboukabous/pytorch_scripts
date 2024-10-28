"""
Trains a PyTorch image classification model.
"""

# Importations
import argparse
import os
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchinfo import summary
from timeit import default_timer as timer
from pathlib import Path
import random
from scripts import data_setup, engine, utils, get_data

# Setup Parser
parser = argparse.ArgumentParser(prog='Transfer Learning Model training', description='Train a TL model on custom data.')
parser.add_argument('--model', type=str, default="efficientnet_b0")
parser.add_argument('--weights', type=str, default="EfficientNet_B0_Weights")
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument('--local', action='store_true')
parser.add_argument('--url', type=str, default="data/dataset")

# Setup HyperParameters
TL_MODEL = getattr(torchvision.models, parser.parse_args().model)
TL_WEIGHTS = getattr(torchvision.models, parser.parse_args().weights)
NUM_EPOCHS = parser.parse_args().epochs
BATCH_SIZE = parser.parse_args().batch_size
LEARNING_RATE = parser.parse_args().lr

IS_LOCAL = parser.parse_args().local
URL = parser.parse_args().url

# Setup directories
if IS_LOCAL:
  train_dir, test_dir = get_data.local_data(URL)
else:
  train_dir, test_dir = get_data.download_data(URL)

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create AUTO transforms
weights = TL_WEIGHTS.DEFAULT # # Get a set of pretrained model weights (default = best)
auto_transforms = weights.transforms() # Get the transforms used to create our pretrained weights automatically

# Create dataLoaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               batch_size=BATCH_SIZE,
                                                                               train_transform=auto_transforms,
                                                                               test_transform=auto_transforms)

# Creating a pretrained model
model = TL_MODEL(weights=weights).to(device)

# Freezing the base model
for param in model.parameters():
  param.requires_grad = False

# Update the classifier head to suit our problem
#model.classifier = nn.Sequential(
#    nn.Dropout(p=0.2, inplace=True),
#    nn.Linear(in_features=1280, out_features=len(class_names))
#).to(device)

# Change the classifier out_features and require_grad parameters
"""classifier_input_features = 0
for idx, m in enumerate(model.modules()):
  if isinstance(m, nn.Linear):
    print(m)
    m.out_features = len(class_names)
    classifier_input_features = m.in_features
    m = nn.Linear(in_features=m.in_features, out_features=len(class_names), bias=True)
    for param in m.parameters():
      param.requires_grad = True

classifier_layer = nn.Linear(in_features=classifier_input_features, out_features=len(class_names), bias=True)
# model = nn.Sequential(model[0:-2], classifier_layer)"""

# Get input features of the last Linear layer and replace it
model = utils.replace_last_linear_layer(model, num_classes=len(class_names))

# Print summary to see the changes done
summary(model=model,
        input_size=(1, 3, auto_transforms.crop_size[0], auto_transforms.crop_size[0]),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])

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

# Save the model results using utils.py
models_dir = "models"
results_dir = "results"
model_name = f"{model.__class__.__name__}_model_{NUM_EPOCHS}_ep_{LEARNING_RATE}_lr_{BATCH_SIZE}_bs#{random.randint(1, 100)}"

utils.save_model(model=model, target_dir=models_dir, model_name=model_name+".pth")
utils.save_results(results=model_results, target_dir=results_dir, model_name=model_name+".csv")
utils.plot_loss_curves(results=model_results, save=True, target_dir=results_dir, model_name=model_name+".png")

# Some test image predictions on test set
IMAGE_SIZE = (auto_transforms.crop_size[0], auto_transforms.crop_size[0])
NUM_IMAGE_TO_PLOT = 3

# Get some random image path from the test set
test_images_path_list = list(Path(test_dir).glob("*/*.jpg"))
test_images_path_samples = random.sample(test_images_path_list, k=NUM_IMAGE_TO_PLOT)

# Make predictions on and plot the images
i = 0
for image_path in test_images_path_samples:
  utils.pred_plot_image(model=model, image_path=image_path, class_names=class_names, transform=auto_transforms, device=device,
                        save=True, target_dir=results_dir, model_name=f"{model_name}_pred_{i}.png")
  i += 1
