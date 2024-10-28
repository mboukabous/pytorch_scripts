"""
Contains various utility functions for PyTorch model training.
"""

import torch
from torch import nn
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt

def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
  """
  Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(), f=model_save_path)

def save_results(results: dict, target_dir: str, model_name: str):
  """Saves a PyTorch model results to a target directory.

  Args:
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model results.

  Example usage:
    save_model(model=model_name,
               target_dir="results",
               model_name="model_name_results")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)

  # Create model results save path
  model_save_path = target_dir_path / model_name

  # Creating a dataframe from results dictionnary
  results_df = pd.DataFrame(results)

  # Save the model state_dict()
  print(f"[INFO] Saving results to: {model_save_path}")
  results_df.to_csv(model_save_path, index=False)

def load_model(model: torch.nn.Module, model_path: str) -> torch.nn.Module:
  """
  Load a PyTorch model parameters from a target path.

  Args:
    model_path: A path for loading the model from.

  Example usage:
    load_model(model_path="models/model_name.pth")
  """
  # Load the model state_dict()
  print(f"[INFO] Loading model from: {model_path}")
  model.load_state_dict(torch.load(model_path, weights_only=False))
  return model

# Plot loss curves
def plot_loss_curves(results, save=False, target_dir=None, model_name=None):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(1, len(results["train_loss"])+1)

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot save
    if save:
      # Create target directory
      target_dir_path = Path(target_dir)
      target_dir_path.mkdir(parents=True, exist_ok=True)

      # Create save path
      save_path = target_dir_path / model_name

      # Save the curves
      print(f"[INFO] Saving results curves to: {save_path}")
      plt.savefig(save_path)

def replace_last_linear_layer(model, num_classes):
    """
    Replaces the last Linear layer in the model with a new Linear layer 
    that has `num_classes` output features.

    Args:
        model: The neural network model.
        num_classes: The number of output features for the new Linear layer.

    Example usage:
        replace_last_linear_layer(model, num_classes=3)
    """
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            # Check if the Linear layer is part of a Sequential container
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent_module = dict(model.named_modules())[parent_name]
                if isinstance(parent_module, nn.Sequential):
                    # Replace the child module in the Sequential container
                    parent_module[int(child_name)] = nn.Linear(in_features, num_classes)
                    print(f"Replaced layer '{name}' within Sequential with a new Linear layer with {num_classes} output features.")
                    # Ensure requires_grad is True for the new layer
                    for param in parent_module[int(child_name)].parameters():
                        param.requires_grad = True
            else:
                # Replace the standalone Linear layer
                setattr(model, name, nn.Linear(in_features, num_classes))
                print(f"Replaced layer '{name}' with a new Linear layer with {num_classes} output features.")
                # Ensure requires_grad is True for the new layer
                for param in getattr(model, name).parameters():
                    param.requires_grad = True
            break
    else:
        print("No Linear layer found to replace.")
    return model
