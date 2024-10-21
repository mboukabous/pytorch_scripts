"""
Makes a prediction on a target image and plots the image
with its prediction using a PyTorch model.
"""

# Importations
import argparse
from pathlib import Path
import requests
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from scripts import model_builder, utils

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup Parser
parser = argparse.ArgumentParser(prog='Model predicting', description='Prediction using a PyTorch model.')

parser.add_argument('--model', type=str, default="models/model.pth")
parser.add_argument('--local', action='store_true')
parser.add_argument('--path', type=str, default="data/img.jpeg")
parser.add_argument('--classes', type=str)

# Setup HyperParameters
MODEL_PATH = parser.parse_args().model
IS_LOCAL = parser.parse_args().local
IMAGE_PATH = parser.parse_args().path
IMAGE_NAME = IMAGE_PATH.split("/")[-1]
CLASSES = parser.parse_args().classes

# Get HyperParameters
HIDDEN_UNITS = int(MODEL_PATH.split("_hu")[0].split("_")[-1])
INPUT_SHAPE = int(MODEL_PATH.split("_cc_")[0].split("_")[-1])
IMG_SIZE = (int(MODEL_PATH.split("_img")[0].split("_")[-1]), int(MODEL_PATH.split("_img")[0].split("_")[-1]))
CLASS_NAMES = CLASSES.split(",") if CLASSES else None
assert CLASS_NAMES != None, "Classes names are required"

# 1. Load in image and convert the tensor values to float32

if IS_LOCAL:
  target_image_path = Path(IMAGE_PATH)
else:
  dataset_path = Path("data/")
  dataset_path.mkdir(parents=True, exist_ok=True)
  target_image_path = dataset_path / IMAGE_NAME

  # Download the image if it doesn't exists
  if not target_image_path.is_file():
    with open(target_image_path, "wb") as f:
      request = requests.get(IMAGE_PATH)
      print(f"Downloading {target_image_path}")
      f.write(request.content)
  else:
    print("The image already exists")

target_image = torchvision.io.read_image(target_image_path).type(torch.float32)

# 2. Divide the image pixel values by 255 to get them between [0, 1]
target_image = target_image / 255.
# 3. Transform if necessary
transform = transforms.Compose([
    transforms.Resize(size=IMG_SIZE)
])
target_image = transform(target_image)

# 4. Create and load a model
model = model_builder.TinyVGG(input_shape=INPUT_SHAPE,
                              hidden_units=HIDDEN_UNITS,
                              output_shape=len(CLASS_NAMES)).to(device)

model = utils.load_model(model=model, model_path=MODEL_PATH)

# 5. Turn on model evaluation mode and inference mode
model.eval()
with torch.inference_mode():
    # Add an extra dimension to the image
    target_image = target_image.unsqueeze(dim=0)

    # Make a prediction on image with an extra dimension and send it to the target device
    target_image_pred = model(target_image.to(device))

# 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

# 7. Convert prediction probabilities -> prediction labels
target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

# 8. Plot the image alongside the prediction and prediction probability
plt.imshow(target_image.squeeze().permute(1, 2, 0)) # make sure it's the right size for matplotlib
if CLASS_NAMES:
    title = f"Pred: {CLASS_NAMES[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
else:
    title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
plt.title(title)
plt.axis(False)

save_path = f"results/{IMAGE_NAME.split('.')[0]}_prediction.png"
plt.savefig(save_path)
print(f"[INFO] Prediction done and saved in: {save_path}")
